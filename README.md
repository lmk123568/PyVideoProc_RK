# PyVideoProc_RK

基于 Rockchip（瑞芯微）的 Python 视频解码推理实现：**MPP 硬解 → RGA 硬转色/缩放 → dma_buf 零拷贝 numpy 视图 → RKNN NPU 推理**。整条链路上 CPU 不搬一个像素，Python 侧 `read()` 拿到的是硬件缓冲的 numpy 视图（`owndata=False`）。

目前实验支持

- RK3588 —— 已在 Orange Pi 5 Plus 实测（内核 5.10.160 BSP 1.2.0，NPU 驱动 0.9.6）
- RK3576 —— 同代 NPU/MPP/RGA，代码路径相同，未实测

---

## 一、两层分工

```
kkk/
├── main.py           ← ★ 多路入口：一路一个进程、一路一个 NPU 核（解码 → 推理 → 画框）
├── yolo.py           ← YOLO26 后处理/NMS（被 main.py import）+ 模型自检（直接跑 = 拿静态图验模型）
├── assets/           ← test.h264 + 模型 + zidane.jpg（自检素材）
├── pyproject.toml / uv.lock / .venv   ← uv 环境（Python 3.10 + numpy / opencv / pybind11 / cmake / ninja）
└── core/             ← 内层：平台相关的东西 —— 移植时整包拷走
    ├── cpp/                  全部 C++ 源码。rk_decoder.cpp 是模块入口（对外 = MppDecoder），
    │                         其余是底层积木：dmabuf / dmabuf_pool / h264_source /
    │                         mpp_decoder / rga_convert（这些 .cpp 直接编进 rk_decoder）
    ├── rknn_rt.py            ctypes 直调 librknnrt.so，替掉 rknn-toolkit2
    ├── build/                cmake 产物 rk_decoder*.so 落在这里（机器相关，不拷）
    ├── third_party/          自带 librga / librockchip_mpp / librknnrt + 头文件
    └── CMakeLists.txt
```

三个 Python 文件，各司其职：

| 文件 | 角色 | 怎么跑 |
|---|---|---|
| `main.py` | **生产入口**：多路解码 + 推理，一路一进程一 NPU 核 | `uv run python main.py --seconds 30` |
| `yolo.py` | **后处理模块**（`CLASSES` / `IN_W,IN_H` / `postprocess`）+ **模型自检** | `uv run python yolo.py` |
| `core/rknn_rt.py` | NPU 运行时封装，被上面两个 import，不单独跑 | `uv run python -c "import sys;sys.path.insert(0,'core');import rknn_rt"` |

**这个划分的意义**：`main.py` / `yolo.py` / `assets/` 是业务层——画框、截图、抽帧参数、后处理阈值、模型素材，改它们不用碰 C++、不用重编；`core/` 是平台层——第三方库、硬件调用、NPU 运行时封装，它自包含，整个目录拷到任何板子都能重建。`assets/` 特意留在外层：换模型/换测试素材是日常操作，不该翻进平台层里。

### 第三方库是怎么带上的

**不是静态打进 `.so` 的** —— `third_party/` 里只有 `.so` 没有 `.a`，静态链接无从谈起。实际做法是：

1. 编译时用 `target_link_directories` 指向 `third_party/{mpp,rga}/lib`
2. 产物 `rk_decoder*.so` 的 RUNPATH 写成 **`$ORIGIN` 相对路径**：
   ```
   RUNPATH = $ORIGIN/../third_party/mpp/lib:$ORIGIN/../third_party/rga/lib
   ```
   `$ORIGIN` = `.so` 自己所在的 `build/`，所以它永远指向**同目录树内**的库

效果等价于"打包成一个文件"：整个 `core/` 拷到哪都能 `import`，不用设 `LD_LIBRARY_PATH`。差别只是模块旁边多躺三个 `.so`。

> 这里有个必须显式设的开关：CMake 默认会往构建树的 RUNPATH 里塞 `target_link_directories` 的**绝对路径**，目录一改名就是死路径（本项目实测踩过）。`core/CMakeLists.txt` 里用 `CMAKE_BUILD_WITH_INSTALL_RPATH ON` 让产物直接采用写死的 `INSTALL_RPATH`。

---

## 二、一帧的通路

```
RTSP/文件 ──ffmpeg(fork, -c copy)──► H.264 裸流
                                      │
                                 MPP 硬解（no-thread 模式）
                                      │  NV12，dma_buf，16 对齐 stride
                                      ▼
                                 RGA 硬件：NV12 → BGR + 缩放（一步完成）
                                      │  写入 dma_buf 池的 slot
                                      ▼
                        有界队列（满则丢最旧，实时语义）
                                      │
                          read() 包成 numpy 视图（capsule 保活）
                                      │  ★ 到这里 CPU 一个像素都没搬过
                                      ▼
              cv2.resize → RGB → RKNN NPU 推理 → 后处理/NMS → 画框
```

三个设计取舍：

- **队列满丢最旧**：推理慢就跳帧追新，不攒旧账（检测要的是"现在"）
- **`read()` 的帧是零拷贝视图**：攥着帧就占着池 slot，上层攥满池时工作线程丢新帧（不阻塞解码器——MPP 输出 buffer 很少）
- **抽帧 `frame_step=N`**：每 N 帧只交付 1 帧。跳的是解码之后的 RGA+入队，**解码本身不能跳**（H.264 的 P 帧依赖前帧，跳了后面全花屏）

---

## 三、移植到新板子

### 3.1 新板子上要有什么

| 需要 | 为什么 | 怎么装 / 怎么验证 |
|---|---|---|
| **内核里的 RGA / MPP / RKNPU / DMA-HEAP 驱动** | 硬件加速全靠它们，用户态库只是壳 | BSP 内核自带，不用装。验证：`ls /dev/rga /dev/mpp_service /dev/dma_heap/` 都在（NPU 走 `/dev/rknpu` 或 `/dev/dri/renderD129`） |
| **`/dev/dma_heap/system-dma32`** | 我们的 dma_buf 池用它分配。**RGA 的 MMU 映射不了 4GB 以上的物理地址**，`system` heap 在 16G 板子上会随机落到高地址 → 偶发 `RGA_BLIT fail`（见第六节） | 内核 `CONFIG_DMABUF_HEAPS_SYSTEM`，BSP 自带。验证：`ls /dev/dma_heap/` 里有 `system-dma32` |
| **`video` 组权限** | `/dev/rga` 是 `crw-rw---- root:video`，非 root 又不在 `video` 组打不开（`/dev/mpp_service` 是 `crw-rw-rw-`，没这问题） | `sudo usermod -aG video $USER` 然后重新登录。验证：`test -r /dev/rga && echo ok` |
| **uv** | 唯一要装的构建工具：Python 解释器、cmake、ninja、pybind11 全由它拉（都是 aarch64 wheel，不碰 apt） | `curl -LsSf https://astral.sh/uv/install.sh \| sh`（需联网；首次 sync 会顺手下 Python 3.10.21，约 90 MB） |
| **g++** | 编译 pybind 模块 | `sudo apt install g++`（或 build-essential） |
| **ffmpeg** | 只有 RTSP 拉流需要（fork 出来 `-c copy` 出 H.264 裸流）；只跑本地 `.h264` 文件不用装 | `sudo apt install ffmpeg` |

**不需要装的**（这是这套东西可移植的关键）：

| 不用装 | 原因 |
|---|---|
| `librga-dev` / `librockchip-mpp-dev` / `librknnrt.so` | `third_party/` 里自带了，构建时用 `$ORIGIN` 相对路径链接，整个目录拷到哪都能 `import`，不用设 `LD_LIBRARY_PATH` |
| `pybind11-dev` | `pyproject.toml` 里的 pip 版 pybind11（≥2.13），顺带避开 apt 那个 2.9 的 `array_t` 重载坑 |
| `python3-dev` | pybind11 的模块在 Linux 上不链 libpython（`readelf -d` 里没有 `libpython`） |
| `cmake` / `ninja` | uv 从 PyPI 装进 `.venv`，不碰 apt |
| `rknn-toolkit2` | 见第四节末，板上推理走 `rknn_rt.py` |

> ⚠️ `third_party/` 里的 `librga` / `librockchip_mpp` 是**用户态库**，必须与目标板内核里的驱动匹配。如果目标板是另一套 BSP，用 `-DUSE_BUNDLED_LIBS=OFF` 改链系统装的（`apt install librga2 librockchip-mpp1`）。

### 3.2 拷贝

`.venv/` 和 `core/build/` 是机器相关的，别拷（`uv sync` 和 cmake 会重建；拷过去反而因为 shebang / 缓存里的绝对路径失效而报错，见第六节）。

```bash
# 在旧板子上：打包。只需要这三样 —— Python 侧 + C++ 侧 + 资源
cd kkk
tar czf rk.tar.gz --exclude=.venv --exclude=build --exclude=__pycache__ \
    main.py yolo.py assets pyproject.toml uv.lock core

# 在新板子上
tar xzf rk.tar.gz
```

包约 17 MB。`uv.lock` 带上，新板子上 `uv sync` 才能装出一模一样的版本。

### 3.3 构建

在 `kkk/` 里跑这三条：

```bash
cd kkk
uv sync
uv run cmake -S core -B core/build -DCMAKE_PREFIX_PATH="$(uv run python -m pybind11 --cmakedir)"
uv run cmake --build core/build -j$(nproc)
```

- `uv sync` 建 `.venv`（Python 3.10.21 + numpy / opencv / pybind11 / cmake / ninja）
- `-DCMAKE_PREFIX_PATH=...` 那句是让 CMake 找到 `.venv` 里的 pybind11，**别省**
- 产物：`core/build/rk_decoder.cpython-310-aarch64-linux-gnu.so`。以后改 C++ 只要重跑第三条

构建完可以顺手验一下可搬迁性（应该只有 `$ORIGIN/...`，没有绝对路径）：

```bash
readelf -d core/build/rk_decoder*.so | grep RUNPATH
```

### 3.4 验证跑通（三步，不用摄像头）

```bash
cd kkk

# ① 解码：应输出 (360, 640, 3) uint8，owndata=False（False = 零拷贝视图）
uv run python -c "
import sys; sys.path.insert(0, 'core/build')
import rk_decoder as rk
d = rk.MppDecoder('assets/test.h264', out_width=640, out_height=360)
d.start(); f = d.read(3000); d.stop()
print('首帧', f.shape, f.dtype, 'owndata=', f.flags.owndata)
"

# ② 推理：应检出 person 0.89 / person 0.84 / tie 0.51
uv run python yolo.py

# ③ 端到端：三路各应跑到 ~17 fps 并有检出
uv run python main.py --file assets/test.h264 --seconds 10
```

三步都过，说明 MPP、RGA、NPU、零拷贝、后处理整条链在新板子上是通的。

### 3.5 换成你自己的流和模型

```bash
uv run python main.py --url rtsp://用户名:密码@摄像头IP/路径    # 换成你的摄像头
uv run python main.py --file your.h264                        # 或本地文件
```

模型换成你自己的：把 `.rknn` 放进 `assets/`，改 `yolo.py` 顶部的 `RKNN_MODEL` 和 `IN_W, IN_H`（**输入尺寸必须和模型一致**，别照文件名猜）。自己转模型见第五节。

---

## 四、多路入口 main.py

```bash
uv run python main.py --help
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--urls` | 无 | 每路的输入 URL，几路就写几个（`--urls a b c`） |
| `--url` | 内置摄像头 | 只给一个地址时，复制成 `--n` 路（验并发用） |
| `--file` | 无 | 本地 `.h264` 裸流，每路各开一份、循环播放（不用摄像头）。相对路径先按 cwd 找，再按 `core/` 找 |
| `--n` | 3 | 开几路。默认 3 正好一路一个 NPU 核 |
| `--model` | `assets/yolo26n_352x640_*.rknn` | 模型路径 |
| `--out-w` / `--out-h` | 640 / 360 | **RGA 解码输出尺寸**。RGA 一步完成转色+缩放，CPU 只碰小图；设 0 则用原始分辨率 |
| `--step` | 1 | 抽帧：每 N 帧取 1 帧（5 = 第 1 帧取、后 4 帧跳） |
| `--snapshot-every` | 5.0 | 每路截图间隔（秒），写 `camN_snapshot.jpg`；0 = 不截 |
| `--seconds` | 0 | 跑多久自动收工；0 = 一直跑，Ctrl+C 停 |

**"每路一个核"是怎么落的**：RK3588 的 NPU 有三个核，三路进程各自 `init_runtime(core_mask=NPU_CORE_0/1/2)`，互不抢核、真正并行。参考的 GPU 版用 `CUDA_VISIBLE_DEVICES` 把进程绑到显卡，NPU 这边**没有等价的环境变量**，`core_mask` 就是那个显式绑核的参数。

进程模型：一路 = 一个 `mp.Process`（spawn，错峰 1s 启动）。子进程 `signal.SIG_IGN` 屏蔽 SIGINT，退出统一由父进程的 `stop_event` 驱动（`--seconds` 和 Ctrl+C 两条路径都不会留残留进程）。

`--step` 怎么选（实测，1920x1080@25 的流、推理约 13 fps 能力）：

| `--step` | 交付 | 处理 | 稳态丢帧 | RGA 负载 | 队列 |
|---|---|---|---|---|---|
| 1 | 25 fps | ~13 fps | 有（追不上就丢） | 25 次/秒 | 常满 |
| 2 | 13.3 fps | 11.9 fps | 25 帧 | 13 次/秒 | 2~3 |
| 5 | 5.3 fps | 5.2 fps | **0** | 5 次/秒 | **恒 0** |

检测任务一般 5 fps 就够看；要追满吞吐用 2。队列恒空 = 拿到的永远是最新帧、没有积压延迟。

### 为什么不用 rknn-toolkit2 跑推理

板上推理走 `rknn_rt.py`，原因都实测过：

1. **依赖直接冲突**：`rknn-toolkit2` 要求 `numpy<=1.26.4`，本项目要 `numpy>=2.2.6`，`uv` 连解都解不出来（`your project and rknn-toolkit2 are incompatible`）
2. 即使解开也要背 torch + onnx + onnxruntime + scipy ≈ **2.5 GB**
3. `rknn-toolkit-lite2` 在 RK3588 上直接报 `Unsupported run platform: Linux aarch64`

`rknn_rt.py` 只依赖自带的 `librknnrt.so` 和标准库 `ctypes`，零额外 Python 包，实测比 toolkit 路径还快（同图 100 次平均：**88.2 ms vs 103.1 ms**）。

---

## 五、模型转换（在 x86 PC 上做，板上不做）

板上只跑推理，**转换需要 rknn-toolkit2，那一步在 PC 上做**（板上装它会和 numpy 版本打架，见上节）。

```
yolo26n.pt ──ultralytics 导出──► yolo26n.onnx ──rknn-toolkit2──► yolo26n_352x640_rk3588_fp16.rknn
```

**YOLO26 的输出格式和 v8 不一样**，后处理不能照抄（`yolo.py` 里已按 YOLO26 实现）：

- 输出 `[1, 84, N]`，84 = 4 + 80；N = 各 stride 头拼接的锚点数（352x640 是 4620 = 44×80 + 22×40 + 11×20）
- 前 4 通道是 **(cx, cy, w, h)，单位是输入图像的像素**，YOLO26 去掉了 DFL 直接回归，**与格子中心无关**，不能按 v8 的 ltrb 相对格子中心解码
- 类分数**已内置 sigmoid**，不要再激活

---

## 六、已知坑

都是实测踩出来的，按症状查：

| 症状 | 根因与解法 |
|---|---|
| 偶发 `RGA improcess 失败` / `RgaBlit fail: Invalid argument`，`dmesg` 里是 `RGA_MMU unsupported memory larger than 4G!` | RGA 的 MMU 只认 4GB 以下物理地址，16G 板子上 `/dev/dma_heap/system` 的分配会随机落到高地址（同一块缓冲地址固定，所以表现为"时好时坏"，跑多路才冒头）。**改用 `/dev/dma_heap/system-dma32`**（`core/cpp/dmabuf.cpp` 已优先它、回退 system）。MPP 自己的 allocator 本来就优先 dma32，所以只有自建池会踩 |
| `/dev/rga` 打不开（`Permission denied`） | 它是 `crw-rw---- root:video`。`sudo usermod -aG video $USER` 后重新登录 |
| 目录改名/拷贝后 `uv run cmake` 报 `Could not find CMAKE_ROOT !!!` | `.venv` 里 console script 的 shebang / 数据目录还指着旧路径，venv 不能整体搬。`rm -rf .venv && uv sync` 重建 |
| 重配时报 `The current CMakeCache.txt is different...` | `build/` 是从别的位置拷来的，缓存里 `CMAKE_HOME_DIRECTORY` 对不上。`rm -rf core/build` 重配 |
| 产物能跑但换台机器/换个目录就 `cannot open shared object file` | RUNPATH 里混进了绝对路径。检查 `readelf -d core/build/rk_decoder*.so \| grep RUNPATH`，应该只有 `$ORIGIN/...`；`core/CMakeLists.txt` 里的 `CMAKE_BUILD_WITH_INSTALL_RPATH ON` 是管这个的 |
| 画面里没目标时，画出**铺满整屏的框**（类别常是 train，分数 0.1~0.57） | 模型对空场景不输出"无检出"，而是吐一个尺寸恰好等于模型输入的退化框。真物体不会同时占满宽和高 → `yolo.py` 的 `MAX_BOX_COVER=0.9` 按"宽高同时≥90%"剔除（设 1.0 关闭）。注意这个跟解码/RGA 无关，纯 cv2 路径同样会出 |
| `RuntimeWarning: invalid value encountered in subtract`，框是 NaN | fp16 模型对空锚点输出 **inf**，`inf - inf = NaN`。`yolo.py` 先按分数筛再算框，并用 `np.isfinite` 显式挡掉 |
| 模型输出框全乱、坐标离谱 | 先确认 `IN_W, IN_H` 和 `.rknn` 的真实输入尺寸一致（读模型元数据，别猜文件名） |
| pybind11 报 `no matching function`（`array_t` 构造） | 2.9 的老坑：`array_t(shape, strides, ptr, capsule)` 里混用 `int` 字面量和 `ssize_t` 会让 `any_container` 推导失败。显式用 `std::vector<py::ssize_t>`；本包用 pip 的 pybind11 ≥2.13 也绕开了 |
| 单路推理开满 3 个 NPU 核没变快 | RKNPU2 不会把**单次**推理拆到多核（实测 102.8 vs 101.4 ms）。要提速得**每路绑一个核**（多进程各用 `NPU_CORE_0/1/2`）并行跑，即 `main.py` 的做法 |
| 摄像头颜色略偏 | 摄像头是 full-range（`yuvj420p`），RGA 默认按 BT.601 limited 转，暗部会被压。检测无影响；要准就显式设色彩空间 |
| ffmpeg 退出时打 `av_interleaved_write_frame(): Broken pipe` | 是"我方主动 kick/关管道"的正常收尾噪音，不是故障。stderr 无缓冲会插在 stdout 前面，看日志别被顺序骗了 |
| 多路时解码线程日志错序 | 同上，stdout 有缓冲、stderr 没有 |
| 抽帧后画面卡顿/花屏 | 检查抽帧是不是写在了解码**之前**——H.264 的 P 帧依赖前帧，解码本身不能跳，只能跳解码之后的转换与入队 |

---

## 七、实测数据（Orange Pi 5 Plus）

| 项目 | 结果 |
|---|---|
| RTSP 硬解 1920x1080@25 | 20 s 解 481 帧、零丢帧；3 路 / 9 路并发各稳 25 fps、RGA 零报错 |
| RGA 一步 NV12→BGR + 缩放到 640x360 | 前处理 1.58 → **0.79 ms/帧**；dma_buf 每 slot 6.22 → **0.69 MB**（省 89%） |
| 推理 yolo26n 352x640 fp16 | 单核 60~103 ms/帧（随负载波动）；`rknn_rt` 88.2 ms vs `rknn-toolkit2` 103.1 ms |
| **多路 main.py（3 路 = 3 进程 = 3 个 NPU 核）** | 三路同拉一条 1080p25 RTSP：各 **19~20 fps** 稳定、各占一核；分项均值 **infer ~43 ms**（瓶颈）/ pre ~4 ms / post ~4 ms / draw ~0.2 ms / wait ~0.1 ms。本地文件源三路各 ~17 fps |
| 端到端（640x360 解码 + 推理） | 文件源 ~12 fps（单路）；摄像头随 `--step` 5.2 ~ 11.9 fps，瓶颈在推理 |
| 抽帧正确性 | `--step 5` 交付的帧与全解的第 0/5/10… 帧**逐字节相同**（最大差 0） |
| 零拷贝 | `read()` 返回的 numpy `owndata=False`，是 dma_buf 视图而非拷贝 |
| 可搬迁性 | `core/` 打包 17 MB，解到全新路径后从零构建 + 三步自检全过；产物 RUNPATH 只有 `$ORIGIN` 相对路径 |

---

## 附：目录对照

```
PyVideoProc_RK/
├── kkk/
│   ├── README.md          ← 本文
│   ├── main.py            ← ★ 多路入口（3 路 = 3 进程 = 3 个 NPU 核）
│   ├── yolo.py            ← 后处理/NMS + 模型自检
│   ├── assets/            ← test.h264 / 模型 / zidane.jpg
│   ├── pyproject.toml  uv.lock  .venv/
│   └── core/              ← ★ 平台层：移植时拷这一个目录
│       ├── cpp/             全部 C++ 源码（rk_decoder.cpp + 底层积木）
│       ├── rknn_rt.py       ctypes 直调 librknnrt.so
│       ├── build/           cmake 产物（机器相关，不拷）
│       ├── third_party/     自带 librga / librockchip_mpp / librknnrt
│       └── CMakeLists.txt
├── zcp/                   ← 阶段 1~3 的原始积木 + 同步门面 demo.py（留档，移植不需要）
├── tests/                 ← 阶段回归测试（依赖根级 zcp/）
├── smoke/                 ← 最早期 C 冒烟测试
└── assets/                ← test.h264 / mobilenet_v1.rknn
```

> `core/cpp/` 只留当前 `core/CMakeLists.txt` 真正在编的 6 个源文件：`rk_decoder.cpp` + dmabuf / dmabuf_pool / h264_source / mpp_decoder / rga_convert。早期阶段的积木（`bindings.cpp` / `nv12_source.*` / 独立 CMakeLists.txt）已从这里删除，原件在根目录 `zcp/` 留档。

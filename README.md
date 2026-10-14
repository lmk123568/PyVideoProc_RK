# PyVideoProc_RK

基于 RK3588 的 Python 多进程多路视频解码推理实现

核心特性：

- MPP 硬解码加速
- RGA 硬件转色缩放加速
- 数据 dma_buf 零拷贝优化
- C++做计算，Python做业务，快速定制开发

🔥🔥🔥 [基于 CUDA 的 Python 多进程多路视频解码推理实现](https://github.com/lmk123568/PyVideoProc)

## Usage

> 💡 在另外 x86 linux 环境导出 rknn 模型
>
> ```bash
> # 将训练的 yolo26 onnx 模型导出 rknn 模型
> # 输入尺寸 352x640 接近视频帧的 9:16
> pip install rknn-toolkit2
> python assets/convert_rknn.py yolo26n_352x640.onnx
> # 本项目自带转好的模型 asstes/yolo26n_352x640_fp16.rknn
> ```
>

RK3588 开发板所需系统为 ubuntu 22，内核版本为 6.x，推荐镜像包下载：[百度网盘](https://pan.baidu.com/s/1wQJADWox-dDGDBWesU4PBA?pwd=gvbg)

搭建环境

```bash
# 安装 mpp rga 驱动
sudo add-apt-repository ppa:liujianfeng1994/rockchip-multimedia
sudo apt update
sudo apt install git build-essential ffmpeg pkg-config librockchip-mpp-dev librga-dev librga2 -y
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env 
```

克隆本仓库

```bash
git clone https://github.com/lmk123568/PyVideoProc_RK.git
```

开启编译

```bash
cd PyVideoProc_RK
# 安装依赖
uv sync
# 编译 c++
uv run cmake -S core -B core/build
uv run cmake --build core/build
# 开启性能模式
bash set_max_perf.sh
```

测试 yolo26 模型

```bash
uv run python yolo26_rk.py
```

测试多路视频解码推理

打开`main.py`，修改添加视频流

```bash
# 运行
uv run python main.py
```

## Benchmark

路数均分给 3 个 npu 核

- 测试模型 `assets/yolo26n_352x640_fp16.rknn`
- 测试视频流 `h264 1980x1280@25fps`
- RK3588 开最高性能

|      | CPU  | NPU                                | RAM     | YOLO Infer     |
| ---- | ---- | ---------------------------------- | ------- | -------------- |
| 3路  | 6%   | Core0: 11%, Core1: 12%, Core2: 12% | 2.46 GB | 36 ms / frame  |
| 6路  | 11%  | Core0: 23%, Core1: 23%, Core2: 24% | 2.72 GB | 68 ms / frame  |
| 9路  | 15%  | Core0: 40%, Core1: 42%, Core2: 41% | 3.02 GB | 89 ms / frame  |
| 12路 | 20%  | Core0: 56%, Core1: 59%, Core2: 57% | 3.29 GB | 110 ms / frame |

## License

MIT
# PyVideoProc_RK

基于 RK3588 的 Python 多进程多路视频解码推理实现

核心特性：

- MPP 硬解码加速
- RGA 硬件转色缩放加速
- 数据 dma_buf 零拷贝优化
- C++做计算，Python做业务，快速定制开发

🔥🔥🔥 [基于 CUDA 的 Python 多进程多路视频解码推理实现](https://github.com/lmk123568/PyVideoProc)

## Usage

推荐使用 ubuntu，镜像包下载：[百度网盘](https://pan.baidu.com/s/1wQJADWox-dDGDBWesU4PBA?pwd=gvbg)

> 💡 提前准备 rknn 模型
>
> 将训练好的 yolo26 检测模型导出 onnx，输出尺寸为 `352x640`，然后在 linux 服务器上做权重转换，参考如下
>
> ```bash
> pip install rknn-toolkit2
> python assets/convert_rknn.py yolo26n_352x640.onnx
> ```
>
> 导出模型，在开发板上跑通`yolo26_rk.py`即验证成功
>
> （本项目已包含 `yolo26n_352x640_fp16.rknn`，可快速测试）

推荐使用 Ubuntu 系统（[下载镜像包](https://pan.baidu.com/s/1wQJADWox-dDGDBWesU4PBA?pwd=gvbg)），运行如下 CLI

```bash
# 安装基本包
sudo apt install git build-essential ffmpeg pkg-config librockchip-mpp-dev librga-dev librga2
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env 
```

克隆本仓库

```bash
git clone https://github.com/lmk123568/PyVideoProc_RK.git
```

搭建 Python 环境，编译 C++ 给 Python 调用

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
- RK3588 锁最高性能

|      | CPU  | NPU                                | RAM     | YOLO Infer   |
| ---- | ---- | ---------------------------------- | ------- | ------------ |
| 3路  | 6%   | Core0: 11%, Core1: 12%, Core2: 12% | 2.46 GB | 48 ms/frame  |
| 6路  | 11%  | Core0: 23%, Core1: 23%, Core2: 24% | 2.72 GB | 73 ms/frame  |
| 9路  | 15%  | Core0: 40%, Core1: 42%, Core2: 41% | 3.02 GB | 98 ms/frame  |
| 12路 | 20%  | Core0: 56%, Core1: 59%, Core2: 57% | 3.29 GB | 125 ms/frame |



## License

MIT
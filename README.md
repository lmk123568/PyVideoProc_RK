# PyVideoProc_RK

<p>
  <img alt="System" src="https://img.shields.io/badge/System-Ubuntu_22.04-E95420?logo=ubuntu&logoColor=white&style=flat-square">
  <img alt="Kernel" src="https://img.shields.io/badge/Kernel-6.x-00add8?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white&style=flat-square">
  <img alt="Toolkit" src="https://img.shields.io/badge/rknn__toolkit2-2.3.2-7c4f9e?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-2ea94f?logo=mitlicense&logoColor=white&style=flat-square">
</p>

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

RK3588 开发板所需系统为 ubuntu 22.04，内核版本为 6.x，推荐镜像包下载：[百度网盘](https://pan.baidu.com/s/1wQJADWox-dDGDBWesU4PBA?pwd=gvbg)

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

|      | CPU  | NPU                                | RAM    | YOLO Infer     |
| ---- | ---- | ---------------------------------- | ------ | -------------- |
| 3路  | 2%   | Core0: 9%, Core1: 10%, Core2: 10%  | 0.9 GB | 35 ms / frame  |
| 6路  | 8%   | Core0: 23%, Core1: 23%, Core2: 24% | 1.0 GB | 53 ms / frame  |
| 9路  | 12%  | Core0: 33%, Core1: 34%, Core2: 34% | 1.2 GB | 74 ms / frame  |
| 12路 | 16%  | Core0: 48%, Core1: 50%, Core2: 51% | 1.5 GB | 98 ms / frame  |

## License
[MIT](./LICENSE)

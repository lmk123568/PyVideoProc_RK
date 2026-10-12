# -*- coding: utf-8 -*-
"""Convert ONNX -> RKNN (RK3588, FP16, no quantization).

输入约定(与 ultralytics 导出的 ONNX 一致):
  - 输入: RGB, uint8 0-255
  - RKNN 配置 mean=0, std=255 => 实际送入模型的是 /255 后的 0-1 浮点
"""
import sys

from rknn.api import RKNN

ONNX = sys.argv[1] if len(sys.argv) > 1 else 'yolov5n_352x640.onnx'
RKNN_OUT = ONNX.replace('.onnx', '_rk3588_fp16.rknn')

rknn = RKNN(verbose=False)

# 模型输入是 0-1 归一化的浮点,所以这里配 mean=0 / std=255,
# 板端喂 uint8 RGB 图即可,RKNN 会自动做 /255。
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform='rk3588',
)

print('--> Loading ONNX model...')
ret = rknn.load_onnx(model=ONNX)
assert ret == 0, 'load_onnx failed'

print('--> Building RKNN model (FP16, do_quantization=False)...')
ret = rknn.build(do_quantization=False)
assert ret == 0, 'build failed'

print('--> Exporting RKNN model...')
ret = rknn.export_rknn(RKNN_OUT)
assert ret == 0, 'export_rknn failed'

rknn.release()
print('Done:', RKNN_OUT)

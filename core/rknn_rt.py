#!/usr/bin/env python3
"""
rknn 运行时薄封装 —— 用 ctypes 直调 third_party/rknn/lib/librknnrt.so

为什么不用官方的 rknn-toolkit2 / rknn-toolkit-lite2：
  - rknn-toolkit2  import 时就拉 torch（rknn_utils 里），要背 torch+onnx+onnxruntime+scipy
                    ≈2.5GB，还把 numpy 钉死在 <=1.26.4
  - rknn-toolkit-lite2  2.3.0/2.3.2 在这块板子上都报 "Unsupported run platform: Linux aarch64"
                    （平台标识、库路径、LIBRKNNRT_PATH、LD_LIBRARY_PATH、python 版本都排除了）
本封装只依赖 vendored 的 librknnrt.so 和标准库 ctypes，零额外 Python 包。

接口刻意做成和官方 rknn.api.RKNN 一致（load_rknn / init_runtime / inference / release），
所以 main.py / yolo.py 只要换 import 那一行。

结构体布局对着 third_party/rknn/include/rknn_api.h 逐字段抄的，
可用 tools/check_struct_layout.py 跟 C 编译器算出来的 sizeof 对拍验证。
"""

import ctypes
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 常量

RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_QUERY_SDK_VERSION = 5

RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_FLOAT16 = 1
RKNN_TENSOR_INT8 = 2
RKNN_TENSOR_UINT8 = 3
RKNN_TENSOR_INT16 = 4
RKNN_TENSOR_UINT16 = 5
RKNN_TENSOR_INT32 = 6
RKNN_TENSOR_UINT32 = 7
RKNN_TENSOR_INT64 = 8

RKNN_TENSOR_NCHW = 0
RKNN_TENSOR_NHWC = 1
RKNN_TENSOR_NC1HWC2 = 2

RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256


# ---------------------------------------------------------------- 结构体

class rknn_input_output_num(ctypes.Structure):
    _fields_ = [
        ("n_input", ctypes.c_uint32),
        ("n_output", ctypes.c_uint32),
    ]


class rknn_tensor_attr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * RKNN_MAX_DIMS),
        ("name", ctypes.c_char * RKNN_MAX_NAME_LEN),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int),
        ("type", ctypes.c_int),
        ("qnt_type", ctypes.c_int),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]


class rknn_input(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_int),
        ("fmt", ctypes.c_int),
    ]


class rknn_output(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


class rknn_sdk_version(ctypes.Structure):
    _fields_ = [
        ("api_version", ctypes.c_char * 256),
        ("drv_version", ctypes.c_char * 256),
    ]


# ---------------------------------------------------------------- 载库

def _find_lib():
    """按 环境变量 → 项目内 third_party → 系统 的顺序找 librknnrt.so"""
    cands = []
    if os.environ.get('LIBRKNNRT_PATH'):
        cands.append(os.environ['LIBRKNNRT_PATH'])
    # 本文件住在 core/，vendored 的库就在同级的 third_party/ 下
    cands.append(os.path.join(HERE, 'third_party', 'rknn', 'lib', 'librknnrt.so'))
    cands.append('/usr/lib/librknnrt.so')
    for p in cands:
        if os.path.exists(p):
            return p
    raise OSError(
        '找不到 librknnrt.so，找过这些位置：\n  ' + '\n  '.join(cands) +
        '\n可以从 https://github.com/airockchip/rknn-toolkit2 的 rknpu2/runtime 下载，'
        '放到 core/third_party/rknn/lib/ 下，或用 LIBRKNNRT_PATH 指定。')


_LIB_PATH = _find_lib()
_lib = ctypes.CDLL(_LIB_PATH)

# aarch64 上 rknn_context 是 uint64_t（rknn_api.h: `#ifdef __arm__` 才用 uint32）
_ctx_t = ctypes.c_uint64

_lib.rknn_init.restype = ctypes.c_int
_lib.rknn_init.argtypes = [ctypes.POINTER(_ctx_t), ctypes.c_void_p, ctypes.c_uint32,
                           ctypes.c_uint32, ctypes.c_void_p]

_lib.rknn_destroy.restype = ctypes.c_int
_lib.rknn_destroy.argtypes = [_ctx_t]

_lib.rknn_query.restype = ctypes.c_int
_lib.rknn_query.argtypes = [_ctx_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]

_lib.rknn_set_core_mask.restype = ctypes.c_int
_lib.rknn_set_core_mask.argtypes = [_ctx_t, ctypes.c_int]

_lib.rknn_inputs_set.restype = ctypes.c_int
_lib.rknn_inputs_set.argtypes = [_ctx_t, ctypes.c_uint32, ctypes.c_void_p]

_lib.rknn_run.restype = ctypes.c_int
_lib.rknn_run.argtypes = [_ctx_t, ctypes.c_void_p]

_lib.rknn_outputs_get.restype = ctypes.c_int
_lib.rknn_outputs_get.argtypes = [_ctx_t, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]

_lib.rknn_outputs_release.restype = ctypes.c_int
_lib.rknn_outputs_release.argtypes = [_ctx_t, ctypes.c_uint32, ctypes.c_void_p]


# ---------------------------------------------------------------- 主类

class RKNN:
    """官方 rknn.api.RKNN 的等价替身（只实现本项目用到的子集）"""

    NPU_CORE_AUTO = 0
    NPU_CORE_0 = 1
    NPU_CORE_1 = 2
    NPU_CORE_2 = 4
    NPU_CORE_0_1 = 3
    NPU_CORE_0_1_2 = 7
    NPU_CORE_ALL = 0xffff

    def __init__(self, verbose=False):
        self.verbose = verbose
        self._ctx = None
        self._model = None          # 模型文件字节，rknn_init 期间必须保活
        self._n_input = 0
        self._n_output = 0
        self._out_attrs = []

    # ------------------------------------------------------------ 内部
    def _log(self, msg):
        if self.verbose:
            print('[rknn_rt]', msg)

    def _query(self, cmd, struct):
        ret = _lib.rknn_query(self._ctx, cmd, ctypes.byref(struct), ctypes.sizeof(struct))
        if ret != 0:
            raise RuntimeError('rknn_query(cmd=%d) 失败，返回 %d' % (cmd, ret))
        return struct

    # ------------------------------------------------------------ 公开接口
    def load_rknn(self, path):
        """把 .rknn 读进内存。返回 0 成功 / -1 失败（对齐官方 API 的返回约定）"""
        if not os.path.isfile(path):
            print('[rknn_rt] 模型文件不存在: %s' % path)
            return -1
        with open(path, 'rb') as f:
            self._model = f.read()
        self._log('load_rknn %s (%d 字节)' % (path, len(self._model)))
        return 0

    def init_runtime(self, target=None, device_id=None, core_mask=NPU_CORE_AUTO):
        """初始化 NPU 上下文。target/device_id 是为对齐官方签名保留的，板上跑用不到"""
        if self._model is None:
            print('[rknn_rt] 还没 load_rknn')
            return -1

        # create_string_buffer 会复制一份可写内存，rknn_init 期间它必须活着
        model_buf = ctypes.create_string_buffer(self._model, len(self._model))
        ctx = _ctx_t(0)
        ret = _lib.rknn_init(ctypes.byref(ctx), model_buf, len(self._model), 0, None)
        if ret != 0:
            print('[rknn_rt] rknn_init 失败，返回 %d' % ret)
            return ret
        self._ctx = ctx

        if core_mask:
            r = _lib.rknn_set_core_mask(self._ctx, core_mask)
            if r != 0:
                print('[rknn_rt] rknn_set_core_mask(0x%x) 失败，返回 %d' % (core_mask, r))
                return r

        io_num = self._query(RKNN_QUERY_IN_OUT_NUM, rknn_input_output_num())
        self._n_input, self._n_output = io_num.n_input, io_num.n_output

        self._out_attrs = []
        for i in range(self._n_output):
            attr = rknn_tensor_attr()
            attr.index = i
            self._query(RKNN_QUERY_OUTPUT_ATTR, attr)
            self._out_attrs.append(attr)

        self._log('init 完成: %d 输入 / %d 输出, core_mask=0x%x, sdk=%s'
                  % (self._n_input, self._n_output, core_mask, self.get_sdk_version()))
        return 0

    def get_sdk_version(self):
        if self._ctx is None:
            return None
        v = self._query(RKNN_QUERY_SDK_VERSION, rknn_sdk_version())
        return v.api_version.decode('utf-8', 'replace')

    def inference(self, inputs, data_format=None):
        """inputs: [ndarray]，目前支持单输入 uint8 NHWC
        返回: [ndarray]，每项形状 = 模型输出张量形状，dtype float32（want_float=1）"""
        if self._ctx is None:
            raise RuntimeError('还没 init_runtime')

        arr = inputs[0]
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        if arr.dtype != np.uint8:
            raise TypeError('输入 dtype 只支持 uint8，收到 %s' % arr.dtype)
        if data_format is not None and data_format != 'nhwc':
            raise ValueError('只支持 data_format="nhwc"，收到 %r' % data_format)

        inp = rknn_input()
        inp.index = 0
        inp.buf = arr.ctypes.data_as(ctypes.c_void_p)
        inp.size = arr.nbytes
        inp.pass_through = 0                      # 交给驱动按模型 mean/std 归一化
        inp.type = RKNN_TENSOR_UINT8
        inp.fmt = RKNN_TENSOR_NHWC

        ret = _lib.rknn_inputs_set(self._ctx, 1, ctypes.byref(inp))
        if ret != 0:
            raise RuntimeError('rknn_inputs_set 失败，返回 %d' % ret)

        ret = _lib.rknn_run(self._ctx, None)
        if ret != 0:
            raise RuntimeError('rknn_run 失败，返回 %d' % ret)

        outs = (rknn_output * self._n_output)()
        for i in range(self._n_output):
            outs[i].index = i
            outs[i].want_float = 1                # 直接要 float32，省得自己解 fp16
            outs[i].is_prealloc = 0

        ret = _lib.rknn_outputs_get(self._ctx, self._n_output, outs, None)
        if ret != 0:
            raise RuntimeError('rknn_outputs_get 失败，返回 %d' % ret)

        try:
            results = []
            for i in range(self._n_output):
                attr = self._out_attrs[i]
                shape = tuple(int(d) for d in attr.dims[:attr.n_dims])
                n = int(np.prod(shape))
                buf = ctypes.cast(outs[i].buf, ctypes.POINTER(ctypes.c_float))
                # 从驱动内存拷出来 —— release 之后那块内存就还给驱动了
                results.append(np.ctypeslib.as_array(buf, shape=(n,)).reshape(shape).copy())
            return results
        finally:
            _lib.rknn_outputs_release(self._ctx, self._n_output, outs)

    def release(self):
        if self._ctx is not None:
            _lib.rknn_destroy(self._ctx)
            self._ctx = None
            self._model = None
            self._out_attrs = []


if __name__ == '__main__':
    print('librknnrt.so =', _LIB_PATH)
    print('sizeof(rknn_input)       =', ctypes.sizeof(rknn_input))
    print('sizeof(rknn_output)      =', ctypes.sizeof(rknn_output))
    print('sizeof(rknn_tensor_attr) =', ctypes.sizeof(rknn_tensor_attr))

"""RKNN NPU 运行时。

自带 librknnrt.so（和本文件同目录）—— NPU 的 userspace 库必须和内核驱动
版本配对，板子上装的那份不一定对，所以自己带一份，整包拷到新板子就能跑。
这和 mpp/rga 相反：那两个用板子自带的（见 core/CMakeLists.txt）。

这样包一层是为了让调用方写 `from core.rknn import RKNN`，
不用写成 `core.rknn.rknn_rt`（重复）。
"""

from .rknn_rt import RKNN  # noqa: F401

__all__ = ["RKNN"]

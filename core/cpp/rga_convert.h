#pragma once
/*
 * a.md 步骤② 的核心：RGA 硬件把 NV12 源变换（缩放 + 转色）写进 dst dma_buf。
 * fd 进、fd 出，CPU 全程不碰像素 —— 真正的"硬件搬运"。
 *
 * 阶段 3 接上 MPP：解码帧的 Y 面是 hstride×vstride（16 对齐，比 w/h 大），
 * UV 面起始偏移 = hstride×vstride，RGA 靠传入正确 stride 自己算 —— 所以
 * 这里只提供带 stride 的版本。
 *
 * 只出 RGB：下游是 NPU 推理 —— ultralytics 导出的模型（.onnx/.rknn）要的就是
 * RGB，转色在 RGA 里顺手做掉，Python 侧拿到帧直接喂 rknn，不用再 cv2.cvtColor
 * 过一遍 CPU。要 BGR（cv2.imshow/imwrite 那套 OpenCV 约定）把 dst_format 换成
 * RK_FORMAT_BGR_888 即可。
 */
bool rga_nv12_to_rgb_strided(int src_fd, int src_w, int src_h,
                             int src_hs, int src_vs,
                             int dst_fd, int dst_w, int dst_h);

/*
 * 启动时把运行库版本打到 stderr（querystring(RGA_VERSION)），方便排查版本漂开。
 * 不再用 imcheckHeader() 做硬配对检查 —— 那是 im2d API 1.9.3+ 才有的，老库
 * 头文件/.so 都没这符号，硬上编译不过。API 够用与否由 CMake 实测把关。
 */
void rga_check_api();

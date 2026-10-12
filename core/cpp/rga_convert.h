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
 * 编译时用的头文件版本，和板上运行库版本是否配对（RGA 自带的 imcheckHeader）。
 * 用 RGA 之前调一次；不配对抛 std::runtime_error。
 *
 * 为什么必须运行时查：CMake 那条检测只能验头文件（编译期），验不了 .so。
 * librga-dev 的 Depends 是空的（librockchip-mpp-dev 有 = 精确 pin，librga 没有），
 * 理论上头和库可以漂开 —— 漂开的后果不是报错而是花屏，所以宁可提前拦。
 */
void rga_check_api();

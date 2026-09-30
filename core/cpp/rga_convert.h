#pragma once
/*
 * a.md 步骤② 的核心：RGA 硬件把 NV12 源变换（缩放 + 转色）写进 dst dma_buf。
 * fd 进、fd 出，CPU 全程不碰像素 —— 真正的"硬件搬运"。
 *
 * 阶段 3 接上 MPP：解码帧的 Y 面是 hstride×vstride（16 对齐，比 w/h 大），
 * UV 面起始偏移 = hstride×vstride，RGA 靠传入正确 stride 自己算 —— 所以
 * 这里只提供带 stride 的版本。
 *
 * 只出 BGR：下游是 OpenCV 系（cv2.imread/imwrite/VideoCapture 全是 BGR 约定），
 * rk_decoder.cpp 拿到的帧直接就能画框/存图。要 RGB 输出（比如喂 mobilenet），
 * 把 dst_format 换成 RK_FORMAT_RGB_888 加个孪生函数即可 —— 根目录 zcp/ 那份
 * 留了 rgb / rgb_strided 两个版本可抄。
 */
bool rga_nv12_to_bgr_strided(int src_fd, int src_w, int src_h,
                             int src_hs, int src_vs,
                             int dst_fd, int dst_w, int dst_h);

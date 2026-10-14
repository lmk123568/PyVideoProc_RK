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
 * 启动时探测运行库可用性（querystring），把版本号缓存起来供汇总日志使用，
 * 本身不打印。不再用 imcheckHeader() 做硬配对检查 —— 那是 im2d API 1.9.3+
 * 才有的，老库头文件/.so 都没这符号，硬上编译不过。API 够用与否由 CMake 实测把关。
 */
void rga_check_api();

/* 缓存的 RGA 运行库版本（如 "v1.10.0_[8]"；rga_check_api 之前调用返回 "?"）。 */
const char *rga_version();

/*
 * 把 RGA 调度核限定到带 IOMMU 的 RGA3（core0|core1），避开无 IOMMU、只能
 * 寻址 4GB 以下的 RGA2。多核板（RK3588）上默认调度可能把任务派给 RGA2，
 * 一旦 dma_buf 落在 4GB 以上就报 "RGA_MMU unsupported memory larger than 4G!"。
 * 必须在任何 improcess 之前、且在真正调用 RGA 的那个线程里调用。
 */
void rga_use_rga3_only();

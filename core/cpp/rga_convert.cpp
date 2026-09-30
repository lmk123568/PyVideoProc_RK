#include "rga_convert.h"

#include <cstdio>
#include <cstring>

#include <rga/im2d.h>
#include <rga/rga.h>

/* 公共骨架：wrap fd → 显式 rect → 同步 improcess。
 * 带真实 stride：MPP 解码帧的 Y 面是 hs×vs（16 对齐），UV 面在 hs*vs
 * 偏移处 —— RGA 拿到 stride 自己算，UV 位置才对。 */
static bool rga_nv12_convert_strided(int src_fd, int src_w, int src_h,
                                     int src_hs, int src_vs,
                                     int dst_fd, int dst_w, int dst_h,
                                     int dst_format)
{
    rga_buffer_t src = wrapbuffer_fd_t(src_fd, src_w, src_h, src_hs, src_vs,
                                       RK_FORMAT_YCbCr_420_SP);   /* NV12 */
    rga_buffer_t dst = wrapbuffer_fd_t(dst_fd, dst_w, dst_h, dst_w, dst_h,
                                       dst_format);
    rga_buffer_t pat;
    memset(&pat, 0, sizeof(pat));

    /* 坑①：rect 必须显式给整幅图。传 {0,0,0,0} = 处理 0×0 区域，"成功"但一个
     * 像素都不写（smoke_rga 实测）。 */
    im_rect sr = { 0, 0, src_w, src_h };
    im_rect dr = { 0, 0, dst_w, dst_h };
    im_rect pr;
    memset(&pr, 0, sizeof(pr));

    /* 坑②：显式传满 10 参。头文件里 C 版是 7 参（第 7 个是 usage），C++ 版是
     * 10 参（第 7 个是 acquire_fence_fd）——少传容易踩语义/歧义。
     * acquire_fence_fd=0：无入站 fence；usage=IM_SYNC：同步等待完成。 */
    IM_STATUS st = improcess(src, dst, pat, sr, dr, pr,
                             /*acquire_fence_fd=*/0,
                             /*release_fence_fd=*/nullptr,
                             /*opt_ptr=*/nullptr,
                             /*usage=*/IM_SYNC);

    /* 坑③：这版 im2d 有两个成功值——同步模式返回 IM_STATUS_SUCCESS(1)，
     * 通用成功是 IM_STATUS_NOERROR(2)，失败为 0 或负数。 */
    if (st != IM_STATUS_SUCCESS && st != IM_STATUS_NOERROR) {
        std::fprintf(stderr, "[cpp] RGA improcess 失败: %s\n", imStrError_t(st));
        return false;
    }
    return true;
}

bool rga_nv12_to_bgr_strided(int src_fd, int src_w, int src_h,
                             int src_hs, int src_vs,
                             int dst_fd, int dst_w, int dst_h)
{
    return rga_nv12_convert_strided(src_fd, src_w, src_h, src_hs, src_vs,
                                    dst_fd, dst_w, dst_h, RK_FORMAT_BGR_888);
}

#include "rga_convert.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>

#include <rga/im2d.hpp>   /* 10 参 C++ 重载在 .hpp 里；.h 只有 C 版 7 参且不会自动 include .hpp */
#include <rga/im2d_common.h>   /* imconfig() */
#include <rga/rga.h>

static std::string g_rga_ver;

/* 探测运行库并把版本缓存下来（不打印）。
 *
 * 为什么不打印：librga 首次调用会自己往 stdout 打一行
 *   rga_api version 1.10.0_[8]
 * 每个进程都会有，和我们的日志混在一起。这里把这次初始化调用期间的 stdout
 * 临时指向 /dev/null 吞掉那行，版本号自己收进来，随 rk_decoder 的汇总行一起输出。
 *
 * 为什么不做"头文件 vs 运行库配对检查"：那套要用 imcheckHeader() +
 * RGA_CURRENT_API_HEADER_VERSION，是 im2d API 1.9.3 才加的；老库里没这符号，
 * 硬上就编译不过。配对检查的硬拦截留给 CMake 的 API 实测。 */
void rga_check_api()
{
    std::fflush(stdout);
    int saved  = ::dup(STDOUT_FILENO);
    int nullfd = ::open("/dev/null", O_WRONLY);
    if (nullfd >= 0)
        ::dup2(nullfd, STDOUT_FILENO);

    const char *ver = querystring(RGA_VERSION);   /* 首次 RGA 调用：库在此初始化 */

    if (nullfd >= 0) {
        std::fflush(stdout);
        ::dup2(saved, STDOUT_FILENO);
        ::close(nullfd);
    }
    if (saved >= 0)
        ::close(saved);

    /* 返回多行，取第一行，再取冒号后的部分（"RGA_api version : v1.10.0_[8]" → "v1.10.0_[8]"） */
    if (ver && *ver) {
        std::string first(ver, std::strcspn(ver, "\n"));
        const char *colon = std::strrchr(first.c_str(), ':');
        std::string val   = colon ? std::string(colon + 1) : first;
        size_t pos = val.find_first_not_of(" \t");
        g_rga_ver = (pos == std::string::npos) ? val : val.substr(pos);
    }
}

const char *rga_version()
{
    return g_rga_ver.empty() ? "?" : g_rga_ver.c_str();
}

/* 把调度核限定到带 IOMMU 的 RGA3（core0|core1）。
 * RK3588 有 3 个 RGA 核：rga3_core0/rga3_core1 各带独立 IOMMU（可映射 >4GB），
 * rga2 无 IOMMU，只能寻址 4GB 以下的物理地址。默认调度可能把任务派给 rga2，
 * 一旦 dma_buf 落在 4GB 以上就报 "RGA_MMU unsupported memory larger than 4G!"。
 * 这里显式把调度核掩码设成 RGA3 两核，绕开 rga2。成功时不打日志（信息并入汇总行）。
 * 注意：imconfig 是进程级全局配置，须在 improcess 之前调用。 */
void rga_use_rga3_only()
{
    IM_STATUS st = imconfig(IM_CONFIG_SCHEDULER_CORE,
                            IM_SCHEDULER_RGA3_CORE0 | IM_SCHEDULER_RGA3_CORE1);
    if (st != IM_STATUS_SUCCESS && st != IM_STATUS_NOERROR)
        std::fprintf(stderr, "[cpp] imconfig 限定 RGA3 失败: %s\n", imStrError_t(st));
}

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

bool rga_nv12_to_rgb_strided(int src_fd, int src_w, int src_h,
                             int src_hs, int src_vs,
                             int dst_fd, int dst_w, int dst_h)
{
    return rga_nv12_convert_strided(src_fd, src_w, src_h, src_hs, src_vs,
                                    dst_fd, dst_w, dst_h, RK_FORMAT_RGB_888);
}

#include "rga_convert.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include <rga/im2d.h>
#include <rga/rga.h>

/* 头文件 vs 运行库，配对检查。放在这个文件里是因为它跟 improcess 的调用
 * 约定是同一件事：头文件版本决定参数语义，库版本决定实际行为，两者漂开
 * 就是"编译过、跑起来花屏"。 */
void rga_check_api()
{
    /* 默认参数就是 RGA_CURRENT_API_HEADER_VERSION —— 即"我编译时用的头文件
     * 版本"，问库认不认。配对返回 SUCCESS/NOERROR，不配对返回
     * IM_STATUS_ERROR_VERSION(-5)。 */
    const IM_STATUS st = imcheckHeader(RGA_CURRENT_API_HEADER_VERSION);
    if (st != IM_STATUS_SUCCESS && st != IM_STATUS_NOERROR) {
        /* 用 std::string 拼而不是定长 buffer + snprintf：中文一个字符 3 字节，
         * 定长很容易悄悄截断（第一版 320 字节就踩了 -Wformat-truncation）。 */
        char hdr[16];
        std::snprintf(hdr, sizeof(hdr), "0x%08x",
                      (unsigned)RGA_CURRENT_API_HEADER_VERSION);
        throw std::runtime_error(
            std::string("RGA 头文件与运行库不配对：imcheckHeader = ")
            + std::to_string((int)st) + " (" + imStrError_t(st) + ")\n"
              "  编译时头文件 API 版本 = " + hdr + "\n"
              "  多半是 librga-dev 和 librga2 版本不一致 —— librga-dev 的 Depends\n"
              "  是空的，apt 不会把两者对齐。\n"
              "  修：sudo apt install --reinstall librga-dev librga2\n"
              "  （或让 pkg-config 指向版本正确的那份）");
    }

    /* 过了也要打一行 —— 因为 imcheckHeader 只拦一个方向（实测）：
     *   传入 0x01090302（编译时的头）  -> 成功
     *   传入 0x02000000（比库新）      -> 成功   ← 放行
     *   传入 0x01000000（比库老）      -> -5     ← 只拦这个
     * 所以"头文件比库新"它查不出来，而 librga-dev 的 Depends 是空的、正好
     * 可能漂成那样。那行日志是这种情况下唯一的线索。
     * querystring 返回多行，只取第一行（"RGA_api version : v1.9.3_[2]"）。 */
    const char *ver = querystring(RGA_VERSION);
    const std::string first(ver, std::strcspn(ver, "\n"));
    std::fprintf(stderr, "[cpp] RGA 就绪：头文件 API 0x%08x，运行库 %s\n",
                 (unsigned)RGA_CURRENT_API_HEADER_VERSION, first.c_str());
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

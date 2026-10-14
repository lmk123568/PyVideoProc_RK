#include "mpp_decoder.h"

#include <rockchip/rk_vdec_cfg.h>

#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <string>

static void cpp_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void cpp_log(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "[cpp] ");
    std::vfprintf(stderr, fmt, ap);
    std::fflush(stderr);
    va_end(ap);
}

MppDecoder::MppDecoder(H264Source &src, Codec codec)
    : src_(src), buf_(512 * 1024)
{
    if (mpp_packet_init(&pkt_, NULL, 0) != MPP_OK)
        throw std::runtime_error("mpp_packet_init 失败");

    int ret = mpp_create(&ctx_, &mpi_);
    if (ret)
        throw std::runtime_error("mpp_create 失败");
    /* 关内部线程必须在 mpp_init 之前 */
    mpi_->control(ctx_, MPP_SET_DISABLE_THREAD, NULL);
    ret = mpp_init(ctx_, MPP_CTX_DEC, codec == HEVC ? MPP_VIDEO_CodingHEVC
                                                    : MPP_VIDEO_CodingAVC);
    if (ret)
        throw std::runtime_error("mpp_init 失败");

    /* split_parse 走 cfg（官方现行做法）：任意块大小直接喂 */
    MppDecCfg cfg = NULL;
    mpp_dec_cfg_init(&cfg);
    mpi_->control(ctx_, MPP_DEC_GET_CFG, cfg);
    mpp_dec_cfg_set_u32(cfg, "base:split_parse", 1);
    mpi_->control(ctx_, MPP_DEC_SET_CFG, cfg);
    mpp_dec_cfg_deinit(cfg);

    // cpp_log("MppDecoder 就绪（%s, no-thread, split_parse）\n",
    //         codec == HEVC ? "HEVC" : "H264");
}

MppDecoder::~MppDecoder()
{
    drop_current();
    if (pkt_)
        mpp_packet_deinit(&pkt_);
    if (ctx_)
        mpp_destroy(ctx_);
}

void MppDecoder::drop_current()
{
    if (frame_) {
        mpp_frame_deinit(&frame_);
        frame_ = nullptr;
    }
}

void MppDecoder::release_current()
{
    drop_current();
}

bool MppDecoder::next_frame()
{
    if (eos_)
        return false;

    int idle = 0;   /* 连续无进展的毫秒数（防死循环的兜底） */

    for (;;) {
        drop_current();   /* 上一帧先还：MPP 只有几个输出 buffer */

        bool have_data = pkt_ && mpp_packet_get_length(pkt_) > 0;
        /* 收尾阶段：零长 EOS 包已发，即便包空了也要继续 decode，
         * 把解码器里攒着的最后几帧 + EOS 帧逼出来 */
        bool draining = src_done_ && eos_sent_;

        if (have_data || draining) {
            MppFrame f = NULL;
            int ret = mpi_->decode(ctx_, pkt_, &f);
            if (ret != MPP_OK)
                throw std::runtime_error("mpi->decode 失败: " + std::to_string(ret));

            if (!f) {
                if (!draining && ++idle > 2000)
                    throw std::runtime_error("解码停滞（数据在却一直不出帧）");
                if (draining && ++idle > 200)
                    return false;   /* 收尾也没帧了：真没了 */
                usleep(1000);
                continue;
            }
            idle = 0;

            if (mpp_frame_get_info_change(f)) {
                /* 格式协商帧：记下参数、回 ACK（不回就死锁）。
                 * 不在此打印——源分辨率由 rk_decoder 的汇总行统一输出。 */
                cur_.width  = mpp_frame_get_width(f);
                cur_.height = mpp_frame_get_height(f);
                cur_.hor_stride = mpp_frame_get_hor_stride(f);
                cur_.ver_stride = mpp_frame_get_ver_stride(f);
                mpi_->control(ctx_, MPP_DEC_SET_INFO_CHANGE_READY, NULL);
                info_acked_ = true;
                mpp_frame_deinit(&f);
                continue;
            }
            if (mpp_frame_get_eos(f)) {
                mpp_frame_deinit(&f);
                eos_ = true;
                cpp_log("MPP 收到 EOS，共 %d 帧\n", n_frames_);
                return false;
            }
            if (mpp_frame_get_errinfo(f)) {
                cpp_log("跳过坏帧 err=%#x\n", mpp_frame_get_errinfo(f));
                mpp_frame_deinit(&f);
                continue;
            }
            /* 正常帧：攥住，摘出 fd/尺寸给上层（fd 在 release 前有效） */
            frame_ = f;
            MppBuffer mb = mpp_frame_get_buffer(f);
            cur_.fd = mpp_buffer_get_fd(mb);
            cur_.width  = mpp_frame_get_width(f);
            cur_.height = mpp_frame_get_height(f);
            cur_.hor_stride = mpp_frame_get_hor_stride(f);
            cur_.ver_stride = mpp_frame_get_ver_stride(f);
            n_frames_++;
            return true;
        }

        /* 包空了 → 从源再读一块（实时流会阻塞在这，天然按帧率节奏） */
        if (!src_done_) {
            size_t n = src_.read(buf_.data(), buf_.size());
            if (n > 0) {
                mpp_packet_set_data(pkt_, buf_.data());
                mpp_packet_set_size(pkt_, n);
                mpp_packet_set_pos(pkt_, buf_.data());
                mpp_packet_set_length(pkt_, n);
                continue;
            }
            src_done_ = true;
        }

        /* 源结束 → 补发零长 EOS 包（触发收尾） */
        if (!eos_sent_) {
            mpp_packet_set_size(pkt_, 0);
            mpp_packet_set_length(pkt_, 0);
            mpp_packet_set_eos(pkt_);
            eos_sent_ = true;
            continue;
        }

        return false;   /* 理论上到不了（draining 分支处理了） */
    }
}

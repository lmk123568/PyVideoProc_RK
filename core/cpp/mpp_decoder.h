#pragma once
/* MPP 硬解封装（no-thread 模式，配方与 smoke/smoke_mpp.c 相同）：
 *   - 所有解码在调用线程内同步完成，无内部线程/队列，生命周期好推理
 *   - 单个会话级 MppPacket 反复重灌（put 后绝不中途 deinit）
 *   - info_change 帧自动回 ACK，split_parse 走 cfg
 *
 * 帧的所有权约定（重要）：next_frame() 返回后，当前帧一直被攥着
 * （fd/尺寸有效），直到下一次 next_frame() 或显式 release_current()。
 * MPP 内部输出 buffer 只有几个，攥着不放解码器就停摆——所以
 * 上层（pool.fill）必须在 RGA 用完 fd 后立刻 release_current()。 */
#include <cstddef>
#include <cstdint>
#include <vector>

#include <rockchip/rk_mpi.h>

#include "h264_source.h"

/* 一帧解码结果：fd 给硬件（RGA），尺寸/stride 给 wrapbuffer */
struct Nv12FrameInfo {
    int fd;
    int width, height;
    int hor_stride, ver_stride;   /* MPP 按 16 对齐，≠ width/height！ */
};

class MppDecoder {
public:
    enum Codec { H264, HEVC };

    MppDecoder(H264Source &src, Codec codec = H264);
    ~MppDecoder();

    MppDecoder(const MppDecoder &) = delete;
    MppDecoder &operator=(const MppDecoder &) = delete;

    /* 阻塞直到解出下一帧；false = 流结束（EOS）。
     * 成功后 current() 有效，直到下次调用。 */
    bool next_frame();
    /* 把当前帧还给解码器（fd 随之失效） */
    void release_current();
    const Nv12FrameInfo &current() const { return cur_; }
    int frame_count() const { return n_frames_; }

private:
    /* 内部：把当前持有的帧 deinit 掉 */
    void drop_current();

    H264Source &src_;
    MppCtx      ctx_  = nullptr;
    MppApi     *mpi_  = nullptr;
    MppPacket   pkt_  = nullptr;
    MppFrame    frame_ = nullptr;    /* 当前攥着的帧 */
    Nv12FrameInfo cur_{};
    std::vector<uint8_t> buf_;       /* 512KB 读缓冲 */
    int   n_frames_ = 0;
    bool  eos_ = false;              /* 收到 EOS 帧，流彻底结束 */
    bool  src_done_ = false;         /* 源读不出数据了 */
    bool  eos_sent_ = false;         /* 零长 EOS 包已发 */
    bool  info_acked_ = false;
};

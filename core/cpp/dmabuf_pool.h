#pragma once
/*
 * cpp::DmaBufPool —— a.md 五步里的 ①④⑤ 的地基
 *
 *   ① 从 /dev/dma_heap/system 分配若干块 dma_buf（"公共货架"）
 *   ④ 每块可被包装成 numpy 视图（"取货卡片"，见 rk_decoder.cpp 的 read()）
 *   ⑤ slot 的生命周期由 capsule 引用链自动管理（release 回池）
 *
 * 线程安全：内部互斥锁，acquire/release 可从任意线程调用。
 */
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "dmabuf.h"

class DmaBufPool {
public:
    /* quiet=true 关掉本池的全部日志（就绪/析构 + 每次 acquire/release）；
     * 后台线程 25fps 场景防刷屏。cpp/ 各模块默认 false，行为不变。 */
    DmaBufPool(int nslots, int width, int height, int channels,
               bool quiet = false);
    ~DmaBufPool();

    DmaBufPool(const DmaBufPool &)            = delete;
    DmaBufPool &operator=(const DmaBufPool &) = delete;

    /* 拿一个空闲 slot；池空返回 -1（非阻塞——接流水线后再换阻塞等待）。
     * 内部已做 invalidate（SYNC START|READ），CPU 随后可安全读。 */
    int  acquire();
    /* 归还 slot；幂等（重复 release 安全）。内部做 flush（SYNC END|WRITE），
     * CPU 期间写过的数据会被刷进物理内存，之后的硬件读能看到。 */
    void release(int slot);

    const DmaBuf &buf(int slot) const { return bufs_[slot]; }

    /* 硬件（RGA/NPU）在 acquire 之后又写了这个 slot 的场景：
     * 把 slot 交给 CPU 读之前必须再 invalidate 一次——acquire 时那次
     * 发生在硬件写之前，防不住后来的 DMA 写入。见 rk_decoder.cpp 的 run_session()。 */
    void begin_cpu_read(int slot);

    int    nslots()   const { return (int)bufs_.size(); }
    int    in_use()   const;
    size_t buf_len()  const { return len_; }
    int    width()    const { return width_; }
    int    height()   const { return height_; }
    int    channels() const { return channels_; }

private:
    int    width_ = 0, height_ = 0, channels_ = 0;
    size_t len_   = 0;
    std::vector<DmaBuf> bufs_;
    std::vector<char>   in_use_;
    std::vector<int>    free_;     /* 空闲 slot 栈（LIFO → 刚释放的优先复用） */
    bool                quiet_ = false;
    mutable std::mutex  mu_;
};

using DmaBufPoolPtr = std::shared_ptr<DmaBufPool>;

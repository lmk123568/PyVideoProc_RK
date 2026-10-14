#include "dmabuf_pool.h"

#include <unistd.h>
#include <sys/mman.h>

#include <cstdio>
#include <stdexcept>
#include <string>

#define ZCP_LOG(...)                                \
    do {                                            \
        std::fprintf(stderr, "[cpp] " __VA_ARGS__); \
        std::fflush(stderr);                        \
    } while (0)

DmaBufPool::DmaBufPool(int nslots, int width, int height, int channels, bool quiet)
    : width_(width), height_(height), channels_(channels),
      len_((size_t)width * height * channels), quiet_(quiet)
{
    if (nslots <= 0 || width <= 0 || height <= 0 || channels <= 0)
        throw std::invalid_argument("nslots/width/height/channels 必须为正");

    bufs_.reserve(nslots);
    in_use_.assign(nslots, 0);
    free_.reserve(nslots);

    for (int i = 0; i < nslots; i++) {
        DmaBuf b = dmabuf_alloc(len_);
        if (b.fd < 0)
            throw std::runtime_error("dma_heap 分配第 " + std::to_string(i) + " 块失败");
        bufs_.push_back(b);
    }

    /* 空闲栈倒序入栈：acquire 从栈顶 pop，所以第一次拿到 slot 0（符合直觉）；
     * release 时 push 回栈顶，复用时"刚释放的优先"（LIFO，对缓存友好） */
    for (int i = nslots - 1; i >= 0; i--)
        free_.push_back(i);

    if (!quiet_)
        ZCP_LOG("pool 就绪: %d slots × %zu B (w=%d h=%d c=%d), vaddr %p..%p\n",
                nslots, len_, width_, height_, channels_,
                bufs_[0].vaddr, bufs_.back().vaddr);
}

DmaBufPool::~DmaBufPool()
{
    for (auto &b : bufs_) {
        if (b.vaddr) munmap(b.vaddr, b.len);
        if (b.fd >= 0) close(b.fd);
    }
    if (!quiet_)
        ZCP_LOG("pool 析构: %zu 块 dma_buf 已归还内核\n", bufs_.size());
}

int DmaBufPool::acquire()
{
    std::lock_guard<std::mutex> lk(mu_);
    if (free_.empty())
        return -1;

    int slot = free_.back();
    free_.pop_back();
    in_use_[slot] = 1;

    /* CPU 即将读这块内存（Python 侧 frame[...]）。
     * 若此前是 RGA/NPU 硬件写的，invalidate 保证 CPU 看到物理内存里的新数据。 */
    dmabuf_read_begin(bufs_[slot].fd);

    if (!quiet_)
        ZCP_LOG("slot %d acquire  vaddr=%p fd=%d  (in_use=%d/%d)\n",
                slot, bufs_[slot].vaddr, bufs_[slot].fd, in_use(), nslots());
    return slot;
}

void DmaBufPool::release(int slot)
{
    std::lock_guard<std::mutex> lk(mu_);
    if (slot < 0 || slot >= (int)bufs_.size() || !in_use_[slot])
        return;   /* 幂等：重复/非法 release 直接忽略 */

    /* CPU 可能写过（Python 侧 frame[...] = x），flush 保证之后的硬件读到新数据 */
    dmabuf_write_end(bufs_[slot].fd);

    in_use_[slot] = 0;
    free_.push_back(slot);

    if (!quiet_)
        ZCP_LOG("slot %d release  (in_use=%d/%d)\n", slot, in_use(), nslots());
}

void DmaBufPool::begin_cpu_read(int slot)
{
    dmabuf_read_begin(bufs_[slot].fd);
}

int DmaBufPool::in_use() const
{
    int n = 0;
    for (char c : in_use_)
        n += c;
    return n;
}

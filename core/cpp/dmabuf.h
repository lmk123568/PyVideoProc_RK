#pragma once
/*
 * dma_buf 小工具：内核分配（dma_heap）+ CPU 访问同步（DMA_BUF_IOCTL_SYNC）
 * —— smoke_rga 踩坑教训的固化，cpp/ 各组件共用。
 */
#include <cstddef>
#include <cstdint>

/* 一块 dma_buf：fd 是给硬件（RGA/NPU）的"通行证"，vaddr 是给 CPU 的窗口 */
struct DmaBuf {
    int    fd    = -1;
    void  *vaddr = nullptr;
    size_t len   = 0;
};

/* 从 /dev/dma_heap/system 分配（内核保证物理连续，适合 RGA/NPU 做 DMA）。
 * 失败时 fd = -1。 */
DmaBuf dmabuf_alloc(size_t len);

/* CPU 访问 dma_buf 前后跟内核"打招呼"（cache 一致性）：
 *   读硬件刚写的内存前 → dmabuf_read_begin（invalidate，丢掉 CPU 里的旧行）
 *   CPU 要写时         → dmabuf_write_begin / dmabuf_write_end 成对使用
 *                        （END 时把 dirty 行刷进物理内存，之后的硬件读才看得见）
 * 不打招呼 = 时好时坏的幽灵 bug（smoke_rga 实测：密集循环 20/20 必现） */
void dmabuf_read_begin(int fd);
void dmabuf_write_begin(int fd);
void dmabuf_write_end(int fd);

#include "dmabuf.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <cstdio>
#include <cstring>

#include <linux/dma-buf.h>
#include <linux/dma-heap.h>

/* 选 heap：RGA 的 MMU 只认 4GB 以下的物理地址，而 16G 板子上
 * /dev/dma_heap/system 的分配会（看内存碎片情况）落到高地址，症状是偶发
 *   [cpp] RGA improcess 失败: Failed to call RockChipRga interface
 *   dmesg: rga_mm: RGA_MMU unsupported memory larger than 4G!
 *          rga_mm: dst channel map job buffer failed!
 * system-dma32 是 Rockchip BSP 的 DMA32 受限 heap（内核 5.10 板子上都在），
 * 分配保证落在低 4G。MPP 自己的 allocator 也是优先用它（librockchip_mpp
 * 里有 "system-dma32" 字符串），所以解码帧侧不踩这个坑。
 * 老内核没有该 heap 时回退 system（那种板子内存小，本来也到不了 4G 以上）。 */
static const char *g_heap_name = nullptr;

static int open_dma_heap()
{
    static const char *kHeaps[] = {
        "/dev/dma_heap/system-dma32",
        "/dev/dma_heap/system",
    };

    for (const char *path : kHeaps) {
        int fd = open(path, O_RDWR);
        if (fd >= 0) {
            /* 记下短名（去掉目录前缀）给汇总日志用，不在此打印 */
            g_heap_name = path + std::strlen("/dev/dma_heap/");
            return fd;
        }
    }
    return -1;
}

const char *dmabuf_heap_name()
{
    return g_heap_name ? g_heap_name : "?";
}

DmaBuf dmabuf_alloc(size_t len)
{
    DmaBuf b;

    int heap = open_dma_heap();
    if (heap < 0) {
        std::perror("open /dev/dma_heap/system-dma32|system");
        return b;
    }

    struct dma_heap_allocation_data req = {
        .len      = len,
        .fd_flags = O_RDWR | O_CLOEXEC,
    };
    if (ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &req) < 0) {
        std::perror("DMA_HEAP_IOCTL_ALLOC");
        close(heap);
        return b;
    }
    close(heap);   /* heap fd 只在分配时用，dmabuf fd 独立存活 */

    b.fd    = (int)req.fd;
    b.vaddr = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, b.fd, 0);
    if (b.vaddr == MAP_FAILED) {
        std::perror("mmap");
        close(b.fd);
        b.fd    = -1;
        b.vaddr = nullptr;
        return b;
    }
    b.len = len;
    return b;
}

void dmabuf_read_begin(int fd)
{
    struct dma_buf_sync s = {
        .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ,
    };
    if (ioctl(fd, DMA_BUF_IOCTL_SYNC, &s) < 0)
        std::perror("DMA_BUF_IOCTL_SYNC(START|READ)");
}

void dmabuf_write_begin(int fd)
{
    struct dma_buf_sync s = {
        .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE,
    };
    if (ioctl(fd, DMA_BUF_IOCTL_SYNC, &s) < 0)
        std::perror("DMA_BUF_IOCTL_SYNC(START|WRITE)");
}

void dmabuf_write_end(int fd)
{
    struct dma_buf_sync s = {
        .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE,
    };
    if (ioctl(fd, DMA_BUF_IOCTL_SYNC, &s) < 0)
        std::perror("DMA_BUF_IOCTL_SYNC(END|WRITE)");
}

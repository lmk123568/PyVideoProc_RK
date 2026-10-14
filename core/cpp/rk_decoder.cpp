/*
 * rk_decoder —— main.py 要的"MPP 硬解 → numpy"后台解码器（pybind11 绑定）
 *
 * 与根目录 zcp/ 的同步门面（根 demo.py，调用者驱动）是两种形态：
 * 这里的 MppDecoder 自带工作线程 + 有界队列，`read()` 立即返回，
 * 专为多进程/多路异步消费设计。接口与参考 .so（mk_api 系，板上缺
 * libmk_api.so 加载不了）保持一致：
 *
 *     dec = rk_decoder.MppDecoder(rtsp_url)     # 或本地 H.264 裸流路径
 *     dec.start()                               # 后台线程：拉流→硬解→RGA
 *     frame = dec.read()                        # (H,W,3) uint8 RGB / None
 *     dec.stop()
 *
 * 常用参数（都是关键字参数，默认值即"不改行为"）：
 *     out_width/out_height=640,360    RGA 一步转色+缩放，CPU 只碰小图
 *     frame_step=5                    抽帧：每 5 帧交付 1 帧（省 RGA/队列开销）
 *
 * 每一帧的通路（全程零 CPU 像素搬运，直到 Python 侧才读）：
 *   ffmpeg(fork) →管道→ MPP no-thread 硬解 → NV12 dma_buf（16 对齐 stride）
 *   → RGA 转色 → RGB dma_buf（池 slot）→ 队列 → read() 包成 numpy 视图
 *
 * 队列语义（实时检测的取舍）：
 *   - 容量 queue_len，满则丢最旧（推理慢就跳帧追新，不攒旧账）
 *   - read() 的帧是 dma_buf 零拷贝视图：攥着帧就占着 slot，
 *     上层把池攥空时工作线程丢新帧（不阻塞解码器——MPP 输出 buffer 很少）
 *
 * 复用 cpp/ 里阶段 3 的全部实测积木：H264Source / MppDecoder / RGA / DmaBufPool。
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "dmabuf_pool.h"
#include "h264_source.h"
#include "mpp_decoder.h"
#include "rga_convert.h"

namespace py = pybind11;

namespace {

void rkd_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void rkd_log(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "[cpp] ");
    std::vfprintf(stderr, fmt, ap);
    std::fflush(stderr);
    va_end(ap);
}

/* numpy 视图的保活链（a.md 步骤⑤）：
 * ndarray 活着 → capsule 活着 → SlotRef 活着 → slot 不回池。
 * SlotRef 持 pool 的 shared_ptr：即使解码器整个 stop()/销毁了，
 * 你手里没放手的帧仍然有效（旧池被引用钉住，到最后一帧释放为止）。 */
struct SlotRef {
    DmaBufPoolPtr pool;
    int           slot;
    ~SlotRef() { pool->release(slot); }
};

} // namespace

class RkDecoder {
public:
    RkDecoder(std::string url, int queue_len, int nslots,
              int out_width, int out_height, int frame_step)
        : url_(std::move(url)), qcap_((size_t)queue_len), nslots_(nslots),
          out_w_(out_width), out_h_(out_height), step_(frame_step)
    {
        if (url_.empty())
            throw std::invalid_argument("url 不能为空");
        if (queue_len < 1)
            throw std::invalid_argument("queue_len 至少为 1");
        if (frame_step < 1)
            throw std::invalid_argument("frame_step 至少为 1（1 = 每帧都取）");
        /* 队列满员 + 正在 RGA 的那块 + 上层手里攥着的余量 */
        if (nslots < queue_len + 2)
            throw std::invalid_argument("nslots 至少要 queue_len + 2");
        /* 输出尺寸要么不指定（0,0 = 原始分辨率），要么两个都给 */
        if ((out_w_ > 0) != (out_h_ > 0))
            throw std::invalid_argument("out_width/out_height 要么都给，要么都给 0");
        if (out_w_ < 0 || out_h_ < 0)
            throw std::invalid_argument("out_width/out_height 不能为负");

        /* 只有真要转色（out_w_ > 0）才用得到 RGA。放在构造函数里而不是解码
         * 线程里：抛出来直接变成 Python 侧的异常，一眼看到；丢进解码线程就
         * 只剩 rga_fail 计数涨，难查。 */
        if (out_w_ > 0)
            rga_check_api();
    }

    ~RkDecoder() { stop(); }

    RkDecoder(const RkDecoder &) = delete;
    RkDecoder &operator=(const RkDecoder &) = delete;

    /* ---------------- 对 Python 的四件套 ---------------- */

    void start()
    {
        std::lock_guard<std::mutex> lk(life_mu_);
        if (th_.joinable())
            return;   /* 已在跑（stop() 末尾会把 th_ 置空线程） */
        stop_flag_ = false;
        running_   = true;
        th_ = std::thread(&RkDecoder::thread_main, this);
        /* 启动日志不在这里打：源分辨率/内存池要等第一帧才知，统一在
         * run_session() 里池就绪时打一行汇总 */
    }

    void stop()
    {
        std::thread t;
        {
            std::lock_guard<std::mutex> lk(life_mu_);
            if (!th_.joinable())
                return;
            stop_flag_ = true;
            /* 踹醒所有睡眠点：等帧的 read()、重连退避的 wait_for */
            cv_.notify_all();
            /* 工作线程可能阻塞在 ffmpeg 管道的 read() 上（网络断最长 10s）。
             * 先拷住 shared_ptr 再 kick：踢的瞬间源对象一定还活着
             * （线程退出会话前不会清 cur_src_）。SIGTERM → 管道 EOF →
             * read() 抛"流结束"→ 会话结束 → 线程看到 stop 退出。 */
            std::shared_ptr<H264Source> src;
            {
                std::lock_guard<std::mutex> sl(src_mu_);
                src = cur_src_;
            }
            if (auto *rtsp = dynamic_cast<RtspH264Source *>(src.get()))
                rtsp->kick();
            t = std::move(th_);
        }
        t.join();   /* 在 life_mu_ 外等：线程退出路径不抢 life_mu_，无死锁 */
        running_ = false;
        /* 队列里残留的 slot 归还各自的池（帧没人取走，主动放回） */
        std::lock_guard<std::mutex> q(qmu_);
        while (!queue_.empty()) {
            QEntry e = queue_.front();
            queue_.pop_front();
            e.pool->release(e.slot);
        }
        pool_.reset();
        rkd_log("解码线程退出，stop: 共解 %d 帧，丢 %d\n",
                frames_.load(), dropped_.load());
    }

    /* 取一帧：None = 暂时没有。timeout_ms>0 时最多等这么久（睡眠期间放掉 GIL） */
    py::object read(int timeout_ms)
    {
        QEntry e{};
        {
            std::unique_lock<std::mutex> lk(qmu_);
            if (queue_.empty() && timeout_ms > 0) {
                py::gil_scoped_release nogil;   // 睡觉别占着 GIL
                cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [this] {
                    return !queue_.empty() || stop_flag_.load();
                });
            }
            if (queue_.empty())
                return py::none();
            e = queue_.front();
            queue_.pop_front();
        }

        /* ---- 把 slot 包成 numpy 视图（a.md 步骤④⑤：shape/strides + capsule） ---- */
        auto *ref  = new SlotRef{e.pool, e.slot};
        auto  caps = py::capsule(ref, [](void *p) { delete (SlotRef *)p; });

        const py::ssize_t h = e.pool->height();
        const py::ssize_t w = e.pool->width();
        /* pybind11 2.9 的坑：花括号里混 int 字面量和 ssize_t 会让重载决议
         * 失败（"no matching function"）→ 必须显式 std::vector<ssize_t> */
        std::vector<py::ssize_t> shape  {h, w, 3};
        std::vector<py::ssize_t> strides{w * 3, 3, 1};

        return py::array_t<std::uint8_t>(
            shape, strides,
            static_cast<std::uint8_t *>(e.pool->buf(e.slot).vaddr),  // data ← dma_buf
            caps);                                                   // base ← 保活链
    }

    /* ---------------- 观测面板 ---------------- */
    int         frame_count() const { return frames_.load(); }
    int         dropped()     const { return dropped_.load(); }
    int         skipped()     const { return skipped_.load(); }
    int         rga_fail()    const { return rga_fail_.load(); }
    bool        is_running()  const { return running_.load(); }
    int         width()       const { return w_.load(); }        /* 输出帧宽 */
    int         height()      const { return h_.load(); }        /* 输出帧高 */
    int         stream_width()  const { return sw_.load(); }     /* 源流宽 */
    int         stream_height() const { return sh_.load(); }     /* 源流高 */
    std::string last_error()
    {
        std::lock_guard<std::mutex> l(err_mu_);
        return last_err_;
    }
    int queue_len()
    {
        std::lock_guard<std::mutex> l(qmu_);
        return (int)queue_.size();
    }

private:
    /* 队列元素：slot 号 + 它所属的池（分辨率中途变化时新旧池共存，
     * 各自的帧各自归还，互不串门） */
    struct QEntry {
        DmaBufPoolPtr pool;
        int           slot;
    };

    void thread_main()
    {
        const bool is_rtsp = url_.rfind("rtsp://", 0) == 0;
        /* 只用带 IOMMU 的 RGA3 核：本板 rga2 无 IOMMU，只能寻址 4GB 以下，
         * 默认调度若派给它，落在 4GB 以上的 dma_buf 会报
         * "RGA_MMU unsupported memory larger than 4G!"。
         * 必须在任何 improcess 之前、且在这个真正调 RGA 的线程里设置。 */
        if (out_w_ > 0)
            rga_use_rga3_only();
        while (!stop_flag_.load()) {
            try {
                run_session(is_rtsp);
            } catch (const std::exception &e) {
                std::lock_guard<std::mutex> l(err_mu_);
                last_err_ = e.what();
                /* 正常收工（stop() 踢断流）时"会话结束"不是故障，不刷长提示；
                 * 只有意外断流才打印诊断信息。 */
                if (!stop_flag_.load())
                    rkd_log("会话结束: %s\n", e.what());
            }
            if (stop_flag_.load())
                break;
            /* 退避后重连（可被 stop() 打断）：RTSP 断流 1s；本地文件播完
             * 0.5s 后从头再放（循环播放，方便拿文件当测试源） */
            std::unique_lock<std::mutex> lk(qmu_);
            cv_.wait_for(lk, std::chrono::milliseconds(is_rtsp ? 1000 : 500),
                         [this] { return stop_flag_.load(); });
        }
    }

    /* 一个"会话" = 建源 + 建解码器 + 一直解到流结束/出错。
     * 会话结束（RTSP 断线、文件播完）后 thread_main 决定重连还是收工。 */
    void run_session(bool is_rtsp)
    {
        std::shared_ptr<H264Source> src =
            is_rtsp
                ? std::static_pointer_cast<H264Source>(std::make_shared<RtspH264Source>(url_))
                : std::static_pointer_cast<H264Source>(std::make_shared<H264FileSource>(url_));
        {
            std::lock_guard<std::mutex> l(src_mu_);
            cur_src_ = src;
        }
        MppDecoder dec(*src, MppDecoder::H264);

        int seen = 0;
        while (!stop_flag_.load() && dec.next_frame()) {
            const Nv12FrameInfo &fi = dec.current();

            /* 抽帧：每 step_ 帧只留 1 帧（第 1 帧留，接着跳 step_-1 帧）。
             * 注意 H.264 的 P 帧依赖前帧，**解码本身不能跳**（跳了后面全是花屏），
             * 这里跳的是解码之后的 RGA 转色 + 入队 —— 省掉这些帧的硬件搬运、
             * 内存带宽和队列翻转（MPP 解一帧才 1~3ms，本来就不是瓶颈）。
             * 放在尺寸判断之前：被跳过的帧不触发建池逻辑。 */
            if (step_ > 1 && (seen++ % step_) != 0) {
                dec.release_current();   /* 必须还帧，否则 MPP 输出 buffer 耗尽停摆 */
                skipped_++;
                continue;
            }

            /* RGA 输出池：第一帧才知道源分辨率 → 惰性创建。
             * 指定了 out_width/out_height 就固定用它（RGA 顺便做缩放，
             * 比如 1920x1080 → 640x360 一次转色+缩放搞定，CPU 不用碰大图）；
             * 没指定则跟随源分辨率，中途换流/协商变化时重建
             * （旧池被队列里的 QEntry 和已发出去的视图钉着，活到最后一帧释放） */
            {
                std::lock_guard<std::mutex> l(qmu_);
                const int pw = out_w_ > 0 ? out_w_ : fi.width;
                const int ph = out_h_ > 0 ? out_h_ : fi.height;
                if (!pool_ || pool_->width() != pw || pool_->height() != ph) {
                    pool_ = std::make_shared<DmaBufPool>(nslots_, pw, ph, 3,
                                                         /*quiet=*/true);
                    w_.store(pw);
                    h_.store(ph);
                    /* 一个进程一行汇总（源分辨率这一刻才确定，所以放这）：
                     * 拉流地址 / 编码+源分辨率 / RGA 输出尺寸 / 抽帧步长 /
                     * RGA 版本 / dma_heap / 内存池 */
                    rkd_log("start: %s, h264 %dx%d, resize%dx%d, step%d, "
                            "rga %s, dma_heap %s, pool %d×%zu B\n",
                            url_.c_str(), fi.width, fi.height,
                            pw, ph, step_, rga_version(), dmabuf_heap_name(),
                            nslots_, (size_t)pw * ph * 3);
                }
                sw_.store(fi.width);
                sh_.store(fi.height);
            }

            /* 先给队列腾位：满了丢最旧（实时语义——追新不追旧） */
            {
                std::lock_guard<std::mutex> l(qmu_);
                if (queue_.size() >= qcap_) {
                    QEntry old = queue_.front();
                    queue_.pop_front();
                    old.pool->release(old.slot);
                    dropped_++;
                }
            }

            int slot = pool_->acquire();   /* 池被上层攥空 → 丢这帧，不堵解码器 */
            if (slot < 0) {
                dec.release_current();
                dropped_++;
                continue;
            }

            bool ok = rga_nv12_to_rgb_strided(fi.fd, fi.width, fi.height,
                                              fi.hor_stride, fi.ver_stride,
                                              pool_->buf(slot).fd,
                                              pool_->width(), pool_->height());
            /* RGA(IM_SYNC) 已完成、MPP 的 fd 已用完 → 立刻还帧，
             * 别堵住解码器的输出 buffer（它只有几个） */
            dec.release_current();
            if (!ok) {
                int n = ++rga_fail_;
                if (n <= 3)   /* 只吼前几次，防刷屏 */
                    rkd_log("RGA 变换失败（第 %d 次，丢帧）。若 dmesg 里是 "
                            "\"RGA_MMU unsupported memory larger than 4G\"，"
                            "说明缓冲落在 4G 以上——池已改用 system-dma32 heap，"
                            "出现这行说明该 heap 不可用\n", n);
                pool_->release(slot);
                dropped_++;
                continue;
            }
            /* RGA 刚 DMA 写完：CPU 读前必须 invalidate（acquire 里那次
             * 发生在写之前防不住）—— 阶段 2 的幽灵 bug 就差这一行 */
            pool_->begin_cpu_read(slot);

            {
                std::lock_guard<std::mutex> l(qmu_);
                queue_.push_back({pool_, slot});
            }
            cv_.notify_one();
            frames_++;
        }

        {
            std::lock_guard<std::mutex> l(src_mu_);
            cur_src_ = nullptr;
        }
    }

    /* ---------------- 状态 ---------------- */
    const std::string url_;
    const size_t      qcap_;
    const int         nslots_;
    const int         out_w_, out_h_;   /* 0 = 不缩放，跟随源分辨率 */
    const int         step_;            /* 抽帧：每 step_ 帧取 1 帧（1 = 全取） */

    std::thread             th_;
    std::atomic<bool>       stop_flag_{false};
    std::atomic<bool>       running_{false};
    mutable std::mutex      life_mu_;      // 只保护 th_ 的换入换出

    std::mutex              qmu_;          // 保护 queue_ / pool_
    std::condition_variable cv_;           // 有帧通知 read()；stop 通知所有人
    std::deque<QEntry>      queue_;
    DmaBufPoolPtr           pool_;         // 当前分辨率的 RGA 输出池

    std::mutex              src_mu_;       // 保护 cur_src_（stop() 要拿它 kick）
    std::shared_ptr<H264Source> cur_src_;

    std::mutex              err_mu_;
    std::string             last_err_;

    std::atomic<int>        frames_{0};    // 解出（含后来被丢的）
    std::atomic<int>        dropped_{0};   // 队列满 / 池空 / RGA 失败
    std::atomic<int>        skipped_{0};   // 抽帧跳过的（frame_step 造成）
    std::atomic<int>        rga_fail_{0};  // 其中 RGA 硬件变换失败（驱动/映射问题）
    std::atomic<int>        w_{0}, h_{0};    // 输出帧尺寸（0 = 还没出帧）
    std::atomic<int>        sw_{0}, sh_{0};  // 源流尺寸（RGA 缩放前）
};

PYBIND11_MODULE(rk_decoder, m)
{
    m.doc() = "MPP 硬解 → RGA NV12→RGB → numpy 零拷贝视图"
              "（后台线程 + 有界队列；复用 cpp/ 里阶段 1/2/3 积木）";

    py::class_<RkDecoder>(m, "MppDecoder")
        .def(py::init<std::string, int, int, int, int, int>(),
             py::arg("url"),
             py::arg("queue_len") = 4,
             py::arg("nslots")    = 6,
             py::arg("out_width")  = 0,
             py::arg("out_height") = 0,
             py::arg("frame_step") = 1,
             "url: rtsp://…（fork ffmpeg 拉流）或本地 H.264 裸流文件路径；"
             "queue_len: 帧队列容量（满则丢最旧）；nslots: dma_buf 池大小"
             "（≥ queue_len+2，攥帧越多要越大）；"
             "out_width/out_height: RGA 输出尺寸（0,0 = 原始分辨率）。"
             "给了就由 RGA 一次完成转色+缩放（如 1920x1080→640x360），"
             "CPU 侧只拿到小图，dma_buf 占用也小得多；"
             "frame_step: 抽帧，每 N 帧只交付 1 帧（1 = 每帧都要）。"
             "如 5 = 第 1 帧交付、后 4 帧跳过。注意跳的只是 RGA 转色+入队，"
             "H.264 的 P 帧依赖前帧所以解码本身不能跳")
        .def("start", &RkDecoder::start, "启动后台解码线程（幂等）")
        .def("stop", &RkDecoder::stop,
             "停止线程 + 清空队列（幂等；已发出去的帧视图仍然有效）")
        .def("read", &RkDecoder::read, py::arg("timeout_ms") = 0,
             "取一帧 (H,W,3) uint8 RGB —— dma_buf 零拷贝视图；None = 暂时没帧。"
             "注意：攥着帧就占着池 slot，全攥满会丢新帧")
        .def_property_readonly("frame_count", &RkDecoder::frame_count,
             "已解出的帧数（含被队列丢弃的）")
        .def_property_readonly("dropped", &RkDecoder::dropped,
             "丢弃数 = 队列满 + 池被攥空 + RGA 失败")
        .def_property_readonly("skipped", &RkDecoder::skipped,
             "抽帧跳过的帧数（frame_step 造成，不是异常）")
        .def_property_readonly("rga_fail", &RkDecoder::rga_fail,
             "其中 RGA 硬件变换失败的次数（>0 说明驱动/内存映射有问题，查 dmesg）")
        .def_property_readonly("queue_len", &RkDecoder::queue_len,
             "当前队列里的帧数")
        .def_property_readonly("is_running", &RkDecoder::is_running,
             "解码线程是否活着")
        .def_property_readonly("width", &RkDecoder::width,
             "输出帧宽（0 = 还没解出第一帧）")
        .def_property_readonly("height", &RkDecoder::height,
             "输出帧高")
        .def_property_readonly("stream_width", &RkDecoder::stream_width,
             "源流宽（RGA 缩放前，如 1920）")
        .def_property_readonly("stream_height", &RkDecoder::stream_height,
             "源流高（如 1080）")
        .def_property_readonly("last_error", &RkDecoder::last_error,
             "最近一次会话异常（排查断流用）");
}

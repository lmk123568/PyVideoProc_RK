#pragma once
/* H.264 Annex-B 字节流源：只管"吐数据块"，不管解码。
 * MPP 的 split_parse 模式对块大小无要求（解析器自己切 NAL），
 * 所以文件和网络流共用这一个极简接口。 */
#include <cstddef>
#include <cstdint>
#include <string>

class H264Source {
public:
    virtual ~H264Source() = default;
    /* 阻塞读一块：返回实际字节数（可能小于 n，实时流常这样），0 = 结束 */
    virtual size_t read(uint8_t *dst, size_t n) = 0;
};

/* 本地裸流文件（assets/test.h264 这类 Annex-B 文件） */
class H264FileSource : public H264Source {
public:
    explicit H264FileSource(const std::string &path);
    ~H264FileSource() override;
    size_t read(uint8_t *dst, size_t n) override;

private:
    int fd_ = -1;
};

/* RTSP 实时流：fork 一个 ffmpeg 子进程拉流，从管道读 H.264。
 * ffmpeg 负责所有脏活（RTSP 握手/鉴权/TCP 交织/断线），我们只拿裸流。
 * URL 里密码带 @ 也没关系：argv 直传不经 shell、不经我们解析，
 * ffmpeg 取最后一个 @ 作 user/host 分隔符（已实测）。 */
class RtspH264Source : public H264Source {
public:
    explicit RtspH264Source(const std::string &url);
    ~RtspH264Source() override;   /* close + SIGTERM + waitpid */
    size_t read(uint8_t *dst, size_t n) override;

    /* 从别的线程"踹"子进程（SIGTERM）：让阻塞在管道 read() 上的工作线程
     * 尽快拿到 EOF 解阻塞（rk_decoder 的 stop() 用）。
     * 与析构里的 SIGTERM 重复无害；调用方须保证此刻对象还活着（拿住引用）。 */
    void kick();

private:
    int   fd_  = -1;   /* 子进程 stdout 的读端 */
    pid_t pid_ = -1;
};

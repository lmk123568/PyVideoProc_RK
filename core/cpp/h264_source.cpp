#include "h264_source.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

/* read(2) 的 EINTR 重试包装：返回真实读到的字节数，0 = EOF */
static ssize_t read_retry(int fd, void *dst, size_t n)
{
    for (;;) {
        ssize_t r = ::read(fd, dst, n);
        if (r >= 0 || errno != EINTR)
            return r;
    }
}

/* ---------------- H264FileSource ---------------- */

H264FileSource::H264FileSource(const std::string &path)
{
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
        throw std::runtime_error("打开 " + path + " 失败: " + std::string(strerror(errno)));
}

H264FileSource::~H264FileSource()
{
    if (fd_ >= 0)
        ::close(fd_);
}

size_t H264FileSource::read(uint8_t *dst, size_t n)
{
    return (size_t)read_retry(fd_, dst, n);
}

/* ---------------- RtspH264Source ---------------- */

RtspH264Source::RtspH264Source(const std::string &url)
{
    int pipefd[2];
    if (::pipe(pipefd) != 0)
        throw std::runtime_error("pipe 失败");

    pid_ = ::fork();
    if (pid_ < 0) {
        ::close(pipefd[0]);
        ::close(pipefd[1]);
        throw std::runtime_error("fork 失败");
    }
    if (pid_ == 0) {
        /* 子进程：stdout 接管道，stderr 继承（ffmpeg 报错直接打到我们终端） */
        ::dup2(pipefd[1], 1);
        ::close(pipefd[0]);
        ::close(pipefd[1]);
        char arg_timeout[] = "10000000";   /* 10s socket IO 超时（ffmpeg 4.4 的
                                              rtsp 叫 -stimeout，微秒；新版本才叫
                                              -rw_timeout，写错名 ffmpeg 直接退出） */
        char *argv[] = {
            (char *)"ffmpeg",
            (char *)"-nostdin",
            (char *)"-loglevel",  (char *)"warning",
            (char *)"-rtsp_transport", (char *)"tcp",
            (char *)"-stimeout",  arg_timeout,
            (char *)"-i",         (char *)url.c_str(),
            (char *)"-map",       (char *)"0:v:0",   /* 只要第一路视频 */
            (char *)"-c",         (char *)"copy",    /* 不重编码，只拆包 */
            /* RTSP 的 SPS/PPS 是带外的（在 SDP 里），-f h264 这个裸流 muxer
             * 不会把它写进输出，MPP 拿不到参数集就一帧都解不出来。dump_extra
             * 只是把 extradata 里的 SPS/PPS 作为前置 NAL 插回每个关键帧前，
             * 不重编码、不碰像素 */
            (char *)"-bsf:v",     (char *)"dump_extra",
            (char *)"-f",         (char *)"h264",    /* Annex-B 裸流 */
            (char *)"pipe:1",
            nullptr};
        ::execvp("ffmpeg", argv);
        _exit(127);   /* exec 失败才会到这 */
    }
    ::close(pipefd[1]);
    fd_ = pipefd[0];
}

void RtspH264Source::kick()
{
    if (pid_ > 0)
        ::kill(pid_, SIGTERM);
}

RtspH264Source::~RtspH264Source()
{
    if (fd_ >= 0)
        ::close(fd_);
    if (pid_ > 0) {
        ::kill(pid_, SIGTERM);
        ::waitpid(pid_, nullptr, 0);
    }
}

size_t RtspH264Source::read(uint8_t *dst, size_t n)
{
    ssize_t r = read_retry(fd_, dst, n);
    if (r == 0)
        throw std::runtime_error(
            "RTSP 流结束（ffmpeg 退出）——检查 URL / 网络 / 摄像头是否在线，"
            "ffmpeg 的报错就打在上面的 stderr 里");
    return (size_t)r;
}

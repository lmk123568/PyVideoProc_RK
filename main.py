#!/usr/bin/env python3
"""
多路解码 + 推理：一路一个进程，一路一个 NPU 核

进程模型照参考 main.py（PyVideoProc 的 GPU 版）：一路视频 = 一个 mp.Process，
进程内串一条"解码 → 推理 → 画框"的流水线，进程之间不共享任何状态。
搬到 RK3588 上只换了两处：

  · 解码：pvp.Decoder      →  core/ 里的 C++ 解码器 rk_decoder.MppDecoder
                               （MPP 硬解 → RGA 转色+缩放 → dma_buf 池 → numpy
                                零拷贝视图；自带后台线程 + 有界队列，read() 立即返回）
  · 推理：pvp.Yolo26DetTRT →  rknn_rt.RKNN（ctypes 直调 librknnrt.so）+ infer.py 的后处理

"每路一个核"落在 RKNN 的 core_mask 上：RK3588 的 NPU 有三个核，三路进程各自
init_runtime(core_mask=NPU_CORE_0/1/2)，互不抢核、真正并行。参考版用
CUDA_VISIBLE_DEVICES 把进程绑到 GPU，NPU 这边没有等价的环境变量，core_mask
就是那个"显式绑核"的参数（见 NPU_CORE_MASK）。

一帧的通路（CPU 全程不碰像素，直到 Python 侧才读）：
  RTSP ──fork ffmpeg──► H.264 裸流 ──MPP 硬解──► NV12(dma_buf)
       ──RGA 硬转色+缩放──► BGR 640x360(dma_buf 池 slot) ──numpy 视图──►
       CPU: resize→640x352 + BGR2RGB ──► NPU 单核推理 ──► 后处理/NMS ──► 画框 + 定时截图

用法（在 kkk/ 里用 uv 的 venv 跑，numpy/opencv 版本才和构建时一致）：
    cd kkk
    uv run python main.py                              # 三路同一条流（验并发；默认隧道摄像头）
    uv run python main.py --file assets/test.h264      # 本地 H.264 裸流，不用摄像头
    uv run python main.py --urls rtsp://a rtsp://b rtsp://c   # 三路各自的源
    uv run python main.py --seconds 30                 # 跑 30 秒自动收工
"""

import argparse
import os
import signal
import sys
import time

import multiprocessing as mp

import cv2
import numpy as np

# 三路日志走同一个 stdout（管道/文件时是块缓冲，父进程那句"已启动 3 路"会憋到
# 进程退出才刷出来，看着像子进程先跑）。行缓冲：谁先写谁先出，时序才是真的。
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, OSError):
    pass

# ---------------------------------------------------------------- 路径
# main.py / yolo.py 住在 kkk/；rknn_rt.py 和解码器编在 core/ 下（core/build/）。
# 模块级就把这几个目录塞进 sys.path —— spawn 出的子进程会重新 import 本文件，这里跟着重跑一遍。
HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(HERE, "core")
sys.path[:0] = [os.path.join(CORE, "build"), CORE, HERE]

# 默认三路用同一条流（阶段 3b 实测可用的隧道摄像头，H.264 1080p25 含音轨）。
# 参考 main.py 是从 cfg.yaml 读每路的 input_url/output_url；这边只改 main.py，
# 配置就直接落在文件里，命令行 --urls / --file 可覆盖。
DEFAULT_URL = "rtsp://admin:Xifu@2021@172.60.34.81/LiveMedia/ch1/Media1"
DEFAULT_MODEL = os.path.join(HERE, "assets", "yolo26n_352x640_rk3588_fp16.rknn")

# NPU 核号 → core_mask（数值同 rknn_rt.RKNN.NPU_CORE_0/1/2；这里写死是为了让
# 主进程不 import rknn_rt（那会连带 dlopen librknnrt.so）也能表达"每路一个核"）
NPU_CORE_MASK = {0: 1, 1: 2, 2: 4}


def draw_dets(img, dets, classes):
    """把检测框画进 img。img 就是解码器发出来的 dma_buf 零拷贝视图 ——
    直接往上面写，不产生拷贝（rtsp_yolo.py 同款）"""
    for c, score, x1, y1, x2, y2 in dets:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "%s:%.2f" % (classes[int(c)], score), (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


class Pipeline(mp.Process):
    """一路视频 = 一个进程：C++ 解码 → NPU 单核推理 → 画框/截图

    与参考 main.py 的 Pipeline 同构（构造函数吃一份 args、run() 里 import 重库），
    区别只在 args 的内容从 gpu/output_url 换成了 npu_core/input_url。
    """

    def __init__(self, args, stop_event):
        super().__init__(name=args["name"])

        self.args = args
        self.stop_event = stop_event

        self.cam = args["name"]
        self.npu_core = args["npu_core"]
        self.input_url = args["input_url"]

    def run(self):
        # 子进程不接 Ctrl+C：统一由主进程 set(stop_event) 通知退出。
        # 否则三路各自可能正好卡在 read()/推理中间被打断，留下半死的解码器。
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # 重库在子进程里 import（参考 main.py 的 `import pvp` 也是放在 run() 里）：
        # 父进程不碰 NPU/RGA/MPP，spawn 出来的子进程各自持有独立的句柄
        try:
            import rk_decoder                                  # C++ 解码器（core/build/）
            from rknn_rt import RKNN                              # NPU（core/rknn_rt.py）
            from yolo import CLASSES, IN_H, IN_W, postprocess     # 后处理/NMS（kkk/yolo.py）
        except ImportError as e:
            print(f"[{self.cam}] 导入失败: {e}\n"
                  f"  解码器要先构建：cd {HERE} && uv sync && "
                  f'uv run cmake -S core -B core/build '
                  f'-DCMAKE_PREFIX_PATH="$(uv run python -m pybind11 --cmakedir)" '
                  f"&& uv run cmake --build core/build -j8")
            return

        # ---- 1. 解码器：C++ 侧后台线程 + 有界队列，read() 立即返回 ----
        # RGA 一步到位：NV12(源尺寸) → BGR(640x360)，转色+缩放都是硬件干，
        # CPU 只碰 640x360 的小图（源 1080p 的大图一张都不进 Python）
        try:
            dec = rk_decoder.MppDecoder(
                self.input_url,
                queue_len=4,                              # 帧队列容量：满则丢最旧（追新不追旧）
                nslots=6,                                 # dma_buf 池大小（≥ queue_len + 2）
                out_width=self.args["out_width"],         # RGA 输出宽（顺便做缩放）
                out_height=self.args["out_height"],       # RGA 输出高
                frame_step=self.args["frame_step"],       # 抽帧：每 N 帧交付 1 帧
            )
        except (RuntimeError, ValueError) as e:
            print(f"[{self.cam}] 解码器创建失败: {e}")
            return

        dec.start()
        print(f"[{self.cam}] 解码启动 pid={os.getpid()} ← {self.input_url} "
              f"输出 {self.args['out_width']}x{self.args['out_height']}"
              + (f"（抽帧 1/{self.args['frame_step']}）" if self.args["frame_step"] > 1 else ""))

        # ---- 2. 等第一帧：既确认流是通的，也顺便拿到真实分辨率 ----
        frame, deadline = None, time.time() + 15
        while frame is None and time.time() < deadline:
            frame = dec.read(1000)
        if frame is None:
            print(f"[{self.cam}] 15 秒没等到帧，退出。最近错误: {dec.last_error()}")
            dec.stop()
            return
        fh, fw = frame.shape[:2]
        print(f"[{self.cam}] 首帧 {fw}x{fh} BGR（源 {dec.stream_width}x{dec.stream_height}），"
              f"owndata={frame.flags.owndata}  ← False = dma_buf 零拷贝视图")
        frame = None          # 放掉这一帧占的 slot，别攥着池去加载模型

        # ---- 3. NPU：模型加载一次，并锁死在这一路的核上 ----
        mask = NPU_CORE_MASK[self.npu_core]
        rknn = RKNN(verbose=False)
        if rknn.load_rknn(self.args["model"]) != 0:
            print(f"[{self.cam}] 模型加载失败: {self.args['model']}")
            dec.stop()
            return
        if rknn.init_runtime(target="rk3588", core_mask=mask) != 0:
            print(f"[{self.cam}] init_runtime 失败（core_mask=0x{mask:x}）")
            dec.stop()
            return
        print(f"[{self.cam}] NPU 就绪 core={self.npu_core} mask=0x{mask:x} "
              f"模型={os.path.basename(self.args['model'])} 输入 {IN_W}x{IN_H}")

        # ---- 4. 主循环：取帧 → 前处理 → 推理 → 后处理 → 画框 ----
        n = 0
        t_start = t_stat = t_save = time.time()
        # 统计窗口内的累加（每个统计点清零，看的是最近 5 秒的稳态）
        sum_wait = sum_pre = sum_infer = sum_post = sum_draw = 0.0
        w_n = 0

        try:
            while not self.stop_event.is_set():
                t0 = time.time()

                # 取一帧：None = 暂时没有（解码线程在重连/池被攥空）。攥着帧就占着
                # 池 slot，所以 frame 用完即弃（下一轮重新绑定就自动还回池里）
                frame = dec.read(500)
                if frame is None:
                    time.sleep(0.005)
                    continue

                t1 = time.time()

                # 前处理：与 infer.py / rtsp_yolo.py 完全同款（resize → BGR2RGB → 加 batch 维）。
                # 解码输出 640x360（16:9，硬件缩放无畸变）→ 模型输入 640x352：最后这步
                # 只剩 8 行的纵向拉伸（~2.2%），CPU 开销可忽略
                img = cv2.resize(frame, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.expand_dims(img, 0)          # [1, H, W, 3] uint8

                t2 = time.time()

                outputs = rknn.inference(inputs=[img], data_format="nhwc")

                t3 = time.time()

                # scale_x/scale_y 把"模型输入坐标"换算回"解码帧坐标"（640x352 → 640x360）
                dets = postprocess(outputs[0], fw / IN_W, fh / IN_H, fw, fh)

                t4 = time.time()

                draw_dets(frame, dets, CLASSES)       # 直接画进 dma_buf 视图，不拷贝

                t5 = time.time()

                n += 1
                w_n += 1
                sum_wait += t1 - t0
                sum_pre += t2 - t1
                sum_infer += t3 - t2
                sum_post += t4 - t3
                sum_draw += t5 - t4

                now = time.time()

                if self.args["snapshot_every"] and now - t_save >= self.args["snapshot_every"]:
                    cv2.imwrite(self.args["snapshot"], frame)
                    top = ", ".join("%s %.2f" % (CLASSES[int(c)], s) for c, s, *_ in dets[:5])
                    print(f"[{self.cam}] 截图 {os.path.basename(self.args['snapshot'])} "
                          f"检出 {len(dets)} 个" + (f"：{top}" if dets else ""))
                    t_save = now

                if now - t_stat >= 5.0:
                    # sum_* 累加的是**秒**，这里换成"每帧毫秒"：k 帧摊到 1000 上
                    k = 1000.0 / max(w_n, 1)
                    print(
                        f"[{self.cam}/core{self.npu_core}] 累计 {n} 帧 | "
                        f"端到端 {w_n / (now - t_stat):.1f} fps | "
                        f"分项均值(ms)：wait {sum_wait * k:.1f} pre {sum_pre * k:.1f} "
                        f"infer {sum_infer * k:.1f} post {sum_post * k:.1f} draw {sum_draw * k:.1f} | "
                        f"解码 交付 {dec.frame_count} 丢 {dec.dropped} 队列 {dec.queue_len}"
                    )
                    t_stat = now
                    sum_wait = sum_pre = sum_infer = sum_post = sum_draw = 0.0
                    w_n = 0

        except KeyboardInterrupt:
            pass                                   # 子进程已屏蔽 SIGINT，这里兜底
        except Exception as e:
            print(f"[{self.cam}] 主循环异常: {type(e).__name__}: {e}")
        finally:
            dec.stop()
            rknn.release()
            dt = time.time() - t_start
            print(f"[{self.cam}] 收工：处理 {n} 帧 / {dt:.1f}s = {n / max(dt, 1e-9):.1f} fps"
                  f"（解码交付 {dec.frame_count}，丢 {dec.dropped}，抽帧跳过 {dec.skipped}）")


def _resolve_file(p):
    """--file 的相对路径：先按 cwd 找，再按 kkk/ 找（测试素材在 kkk/assets/）"""
    if os.path.isabs(p) or os.path.exists(p):
        return p
    alt = os.path.join(HERE, p)
    return alt if os.path.exists(alt) else p


def build_streams(a):
    """每路一份 args（参考 main.py 里 args 是个 dict 的形态保持一致）"""
    if a.file:
        urls = [_resolve_file(a.file)] * a.n       # 本地文件当源：每路各开一份，循环播放
    elif a.urls:
        urls = a.urls                              # 几路就写几个 URL
    else:
        urls = [a.url] * a.n                       # 只给一个 URL：复制成 n 路（验并发用）

    return [{
        "name": f"cam{i + 1}",
        "npu_core": i % len(NPU_CORE_MASK),        # 核按路轮转：0,1,2,0,1,2…（三路正好一人一个）
        "input_url": url,
        "model": a.model,
        "out_width": a.out_w,
        "out_height": a.out_h,
        "frame_step": a.step,
        "snapshot": os.path.join(HERE, f"cam{i + 1}_snapshot.jpg"),
        "snapshot_every": a.snapshot_every,
    } for i, url in enumerate(urls)]


def parse_args():
    ap = argparse.ArgumentParser(
        description="多路解码 + YOLO26 推理（一路一个进程，一路一个 NPU 核）")
    ap.add_argument("--urls", nargs="+", help="每路的输入 URL（几路就写几个）")
    ap.add_argument("--url", default=DEFAULT_URL, help="只给一路地址时，复制成 --n 路")
    ap.add_argument("--file", help="本地 H.264 裸流（给了就每路都用它，循环播放，不用摄像头）")
    ap.add_argument("--n", type=int, default=3, help="开几路（默认 3：正好一路一个 NPU 核）")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=".rknn 模型路径")
    ap.add_argument("--out-w", type=int, default=640, help="RGA 解码输出宽（0=跟随源分辨率）")
    ap.add_argument("--out-h", type=int, default=360, help="RGA 解码输出高")
    ap.add_argument("--step", type=int, default=1, help="抽帧：每 N 帧只取 1 帧（1=全取）")
    ap.add_argument("--snapshot-every", type=float, default=5.0, help="每路截图间隔（秒，0=不截）")
    ap.add_argument("--seconds", type=float, default=0, help="跑多久后自动停（0=一直跑，Ctrl+C 停）")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    streams = build_streams(args)

    # spawn：子进程重新 import 本文件，不吃父进程状态（参考 main.py 同款）。
    # 代价是 Process 的参数必须可 pickle —— 这里传的是 dict + Event，没别的
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    stop_event = mp.Event()
    process_pool = []
    for s in streams:
        vp = Pipeline(s, stop_event)
        vp.start()
        time.sleep(1)          # 错峰启动：三路同时抢 ffmpeg/RGA/dma_buf 容易互相踩
        process_pool.append(vp)

    print(f"已启动 {len(process_pool)} 路："
          + ", ".join(f"{s['name']}(NPU core{s['npu_core']})" for s in streams)
          + "。Ctrl+C 停止全部")

    t0 = time.time()
    try:
        while any(p.is_alive() for p in process_pool):
            time.sleep(1)
            if args.seconds and time.time() - t0 >= args.seconds:
                print(f"已跑满 {args.seconds}s，通知子进程退出…")
                stop_event.set()
                break
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，通知子进程优雅退出…")
        stop_event.set()

    for p in process_pool:
        p.join(timeout=10)

    # 超时没退干净的强杀。注意用 enumerate —— demo.py 里 `for i, p in processes`
    # 直接对 Process 解包抛异常，超时的子进程就永远等不到 terminate
    for i, p in enumerate(process_pool):
        if p.is_alive():
            print(f"[cam{i + 1}] 响应超时，强制终止！")
            p.terminate()

    print("✅ 全部退出")

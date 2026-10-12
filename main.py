import multiprocessing as mp
import time

import numpy as np

from yolo26_rk import yolo26_postprocess


class Pipeline(mp.Process):
    def __init__(self, args):
        super().__init__()

        self.args = args

    def run(self):

        from core.build import rk_decoder
        from core.rknn import RKNN

        dec = rk_decoder.MppDecoder(
            self.args["input_url"],
            queue_len=4,  # 帧队列容量，满则丢最旧
            nslots=6,  # dma_buf 池大小(≥ queue_len + 2)
            out_width=640,  # RGA 输出宽
            out_height=352,  # RGA 输出高
            frame_step=self.args["frame_step"],  # 抽帧，每 5 帧交付 1 帧
        )
        dec.start()  # 启动后台解码线程

        rknn = RKNN(verbose=False)
        rknn.load_rknn(self.args["model"])
        NPU_CORE_MASK = {0: 1, 1: 2, 2: 4}
        rknn.init_runtime(target="rk3588", core_mask=NPU_CORE_MASK[self.args["npu"]])

        frame_count = 0
        sum_wait = 0
        sum_det = 0
        sum_track = 0
        sum_event = 0

        while 1:
            t0 = time.time()

            frame = dec.read(500)

            if frame is None:  
               time.sleep(1)        # 队列空/正在重连, 还没出帧, 接着等
               continue

            frame_count += 1

            t1 = time.time()

            # frame RGB (352, 640, 3) --> (1, 352, 640, 3)
            frame = np.expand_dims(frame, 0)
            outputs = rknn.inference(inputs=[frame], data_format="nhwc")

            det_results = yolo26_postprocess(outputs[0])

            t2 = time.time()

            # 目标跟踪
            # track_results = bytetrack(det_results)

            t3 = time.time()

            # 事件判断
            # car(track_results)
            # person(track_results)

            t4 = time.time()

            sum_wait += t1 - t0
            sum_det += t2 - t1
            sum_track += t3 - t2
            sum_event += t4 - t3

            if frame_count == 1000:
                print(
                    f"[{time.strftime('%m/%d/%Y-%H:%M:%S', time.localtime())}] {self.args['input_url']}, "
                    f"Det: {sum_det:.2f}ms, "
                    f"Track: {sum_track:.2f}ms, "
                    f"Event: {sum_event:.2f}ms, "
                    f"Wait: {sum_wait:.2f}ms "
                )
                frame_count = 0
                sum_det = 0
                sum_track = 0
                sum_event = 0
                sum_wait = 0


if __name__ == "__main__":

    # 可单独弄 config yaml 文件
    args = [
        {
            "npu": 0,
            "frame_step": 5,
            "model": "yolo26n_352x640_rk3588_fp16.rknn",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
        },
        {
            "npu": 1,
            "frame_step": 5,
            "model": "yolo26n_352x640_rk3588_fp16.rknn",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
        },
        {
            "npu": 2,
            "frame_step": 5,
            "model": "yolo26n_352x640_rk3588_fp16.rknn",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
        },
    ]

    mp.set_start_method("spawn")
    process_pool = []

    for i in args:
        vp = Pipeline(i)
        vp.start()
        time.sleep(1)
        process_pool.append(vp)

    for vp in process_pool:
        vp.join()

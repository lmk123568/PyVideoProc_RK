# YOLO26 后处理 + 模型正确性验证
#
# 这个文件有两个身份：
#   1. 被 main.py import 的"后处理模块"—— CLASSES / IN_W / IN_H / postprocess()
#   2. 直接 `python yolo.py` 跑的"模型自检"—— 拿一张静态图过一遍，打印耗时、
#      输出形状、检出框，并把画框结果存成 png，用来确认模型和板子都是好的
#
# 模型信息（从 .rknn 文件头元数据读出）:
#   输入: 1x3x352x640 (H=352, W=640), RGB, mean=0/std=255
#         -> 传 uint8 [1,352,640,3], toolkit 自动 /255 归一化
#   输出: 单个 output0 [1, 84, 4620]
#         84  = 4 (cx,cy,w,h 像素单位) + 80 (类分数, 已内置 sigmoid)
#         4620 = 44*80 + 22*40 + 11*20 (stride 8/16/32 三个 head 沿 anchor 维拼接)
#                (输入 352x640 时 352/8=44, 640/8=80, 依此类推)
#
# 推理走 rknn_rt（ctypes 直调 core/third_party/rknn/lib/librknnrt.so），
# 不用 rknn-toolkit2 —— 它要 numpy<=1.26.4（与本项目 numpy>=2.2.6 直接冲突），
# 还要背 torch/onnx ≈2.5GB。接口与官方 rknn.api.RKNN 一致，只换 import 那一行。

import os
import sys
import time

import numpy as np
import cv2

# rknn_rt.py 住在 core/ 下（平台相关的那一层）。被 main.py import 时它已经把
# core/ 加进 sys.path 了，但 yolo.py 自己也能直接跑，所以这里自己补一次。
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'core'))

from rknn_rt import RKNN

# 资源按本文件所在目录（kkk/）解析，不按 cwd —— 这样从任何地方跑都能找到
# assets/。绝对路径原样返回。
_ASSETS = os.path.join(_HERE, 'assets')


def _asset(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(_ASSETS, p))


RKNN_MODEL = _asset('yolo26n_352x640_rk3588_fp16.rknn')
IMG_PATH = _asset('zidane.jpg')
OUT_PATH = _asset('zidane_result.png')

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']

IN_W, IN_H = 640, 352
CONF_THRESH = 0.25     # 置信度阈值
NMS_THRESH = 0.45      # NMS IOU 阈值
MAX_DET = 300          # 先按分数只留前 300 个再进 NMS (ultralytics 默认同值)
# 退化框上限: 宽和高都 >= 这个比例(占画面)的框直接丢掉。
# 实测隧道空场景(画面里没车): 模型不输出"无检出", 而是吐一个铺满整个输入的
# train 框(框尺寸恒 = 输入尺寸, 分数随时间从 0.10 漂到 0.57), 裁剪后正好盖住
# 整个画面 —— 用户看到的就是"框太大把画面全包了"。真物体不会同时占满宽和高
# (同场景里真实的车只占 2%~17%), 所以这个判据能干净地分开。
# 设成 1.0 即关闭该过滤。
MAX_BOX_COVER = 0.9


def nms(boxes, scores, iou_thresh):
    """
    矩阵化贪心 NMS
    boxes: [N,4] xyxy, 一次广播算完 N×N IoU 矩阵, 贪心循环里只剩整行布尔掩码, 不再逐轮重算 IoU.
    返回保留索引 (分数降序)
    """
    n = boxes.shape[0]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # 一次算完全量 IoU: [N,1] 对 [1,N] 广播 -> [N,N]
    xx1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    yy1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    xx2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    yy2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    iou = inter / (areas[:, None] + areas[None, :] - inter)

    order = np.argsort(-scores)             # 分数降序
    suppressed = np.zeros(n, dtype=bool)
    keep = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= iou[i] > iou_thresh   # 保留 i, 第 i 行整批掩掉高重叠候选
    return keep


def postprocess(out, scale_x, scale_y, img_w, img_h, stats=None):
    """out: [1,84,4620] -> 检测框列表 [(class_id, score, x1,y1,x2,y2 原图坐标)]
    前四通道 = (cx, cy, w, h), 输入图像素单位, 与格子中心无关 (YOLO26 无 DFL 直接回归)
    scale_x/scale_y: 模型输入坐标 → 原图坐标的缩放比 (原图宽/IN_W, 原图高/IN_H)
    stats: 可选 dict, 回填诊断计数 {'cand': 裁剪后候选框数, 'degenerate': 剔除的整屏框数}"""
    feat = out[0]                 # [84, 4620]
    cxcywh = feat[:4, :]          # cx, cy, w, h (输入图像素)
    cls = feat[4:, :]             # 已 sigmoid 的类分数 [80, 4620]

    cls_id = np.argmax(cls, axis=0)                  # [4620]
    cls_max = cls[cls_id, np.arange(cls.shape[1])]   # [4620]
    mask = cls_max > CONF_THRESH
    # 模型对"空锚点"会输出 inf（fp16 量化副产物，实测板上必现）：
    # inf - inf 会冒 RuntimeWarning 且框是 NaN，先显式筛掉；
    # 先筛后算还有个附带好处——坐标只对命中锚点算，省掉 4620 次全量运算
    mask &= np.isfinite(cxcywh).all(axis=0)
    if not mask.any():
        return []

    cx, cy, w, h = cxcywh[:, mask]
    x1 = (cx - w / 2) * scale_x
    y1 = (cy - h / 2) * scale_y
    x2 = (cx + w / 2) * scale_x
    y2 = (cy + h / 2) * scale_y

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    scores = cls_max[mask]
    class_ids = cls_id[mask]

    # ---- 边界裁剪: 模型对贴边物体允许框略溢出图像, clip 回 [0,img_w]x[0,img_h]
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, img_w)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, img_h)

    # 裁剪后剔除面积为 0 的退化框 (整框原本就在图外, 被裁没了)
    ok = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

    # 剔除"整屏框"(见 MAX_BOX_COVER 注释): 宽和高同时占满才丢, 只要有一边
    # 没占满就可能是真物体(比如贴脸的车只占满宽度)
    n_cand = int(ok.sum())
    if MAX_BOX_COVER < 1.0:
        cover_w = (boxes[:, 2] - boxes[:, 0]) / img_w
        cover_h = (boxes[:, 3] - boxes[:, 1]) / img_h
        ok &= (cover_w < MAX_BOX_COVER) | (cover_h < MAX_BOX_COVER)

    if stats is not None:
        stats['cand'] = stats.get('cand', 0) + n_cand
        stats['degenerate'] = stats.get('degenerate', 0) + n_cand - int(ok.sum())

    boxes, scores, class_ids = boxes[ok], scores[ok], class_ids[ok]
    if boxes.shape[0] == 0:
        return []

    # ---- TopK 截断, 限定 N×N IoU 矩阵规模 (300² 才 720KB)
    if boxes.shape[0] > MAX_DET:
        top = np.argsort(-scores)[:MAX_DET]
        boxes, scores, class_ids = boxes[top], scores[top], class_ids[top]

    # ---- 类间隔离: 每类的框沿 x 平移 (类id+1) * 4*img_w, 类间永不相交 → IoU 恒 0,
    # 于是"全类一趟 NMS"与"逐类各自 NMS"数学等价 (torchvision batched_nms 同款技巧)
    shifted = boxes.copy()
    off = (class_ids.astype(np.float64) + 1) * (img_w * 4)
    shifted[:, 0] += off
    shifted[:, 2] += off
    keep = nms(shifted, scores, NMS_THRESH)

    return [(class_ids[k], scores[k], *boxes[k]) for k in keep]


def main():
    """模型自检：静态图 → 推理 → 后处理 → 画框存图。跑通了说明模型和板子都没问题。"""
    orig = cv2.imread(IMG_PATH)
    assert orig is not None, 'read %s failed' % IMG_PATH
    img_h, img_w = orig.shape[:2]

    # 前处理：原图 resize 到模型输入（16:9 → 16:9，无畸变）→ RGB → 加 batch 维
    img = cv2.resize(orig, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)   # [1,352,640,3] uint8

    print('* 模型:', RKNN_MODEL)
    rknn = RKNN(verbose=False)
    assert rknn.load_rknn(RKNN_MODEL) == 0
    assert rknn.init_runtime(target='rk3588', core_mask=RKNN.NPU_CORE_0) == 0

    print('* 推理中…')
    t0 = time.time()
    for _ in range(100):
        outputs = rknn.inference(inputs=[img], data_format='nhwc')
    t1 = time.time()
    rknn.release()
    print('* 单帧耗时: %.1f ms' % ((t1 - t0) * 10))
    print('* 输出形状:', outputs[0].shape)

    dets = postprocess(outputs[0], img_w / IN_W, img_h / IN_H, img_w, img_h)
    print('* NMS 后检出: %d' % len(dets))

    for c, score, x1, y1, x2, y2 in dets:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig, '%s:%.2f' % (CLASSES[int(c)], score), (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        print('  %s %.2f  box(%d,%d,%d,%d)' % (CLASSES[int(c)], score, x1, y1, x2, y2))

    cv2.imwrite(OUT_PATH, orig)
    print('* 已保存:', OUT_PATH)


if __name__ == '__main__':
    main()

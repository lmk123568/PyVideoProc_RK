import time

import numpy as np
import cv2

from core.rknn import RKNN


RKNN_MODEL = 'assets/yolo26n_352x640_rk3588_fp16.rknn'
IMG_PATH = 'assets/zidane.jpg'
OUT_PATH = 'assets/zidane_result.png'

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']


def nms(boxes, scores, iou_thresh):
    """矩阵化贪心 NMS"""
    n = boxes.shape[0]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # 一次算完全量 IoU: [N,1] 对 [1,N] 广播 -> [N,N]
    xx1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    yy1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    xx2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    yy2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    iou = inter / (areas[:, None] + areas[None, :] - inter)

    order = np.argsort(-scores)  # 分数降序
    suppressed = np.zeros(n, dtype=bool)
    keep = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= iou[i] > iou_thresh  # 保留 i, 第 i 行整批掩掉高重叠候选
    return keep


def yolo26_postprocess(out, img_h=352, img_w=640, conf_thre=0.25, nms_thre=0.45, max_det=300):
    """
    out: [1,84,4620] -> [(x1, y1, x2, y2, class_id, score), ...]
    """
    feat = out[0]                 # [84, 4620]
    cxcywh = feat[:4, :]          # cx, cy, w, h (输入图像素)
    cls = feat[4:, :]             # sigmoid 后类分数 [80, 4620]

    cls_id = np.argmax(cls, axis=0)                  
    cls_max = cls[cls_id, np.arange(cls.shape[1])]  
    mask = cls_max > conf_thre

    # 去除 inf (rknn 只支持 int8/fp16)
    mask &= np.isfinite(cxcywh).all(axis=0)
    if not mask.any():
        return []

    cx, cy, w, h = cxcywh[:, mask]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    scores = cls_max[mask]
    class_ids = cls_id[mask]


    # TopK 截断
    if boxes.shape[0] > max_det:
        top = np.argsort(-scores)[:max_det]
        boxes, scores, class_ids = boxes[top], scores[top], class_ids[top]


    # 边界裁剪 clip [0, IN_W] x [0, IN_H]
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, img_w)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, img_h)


    # NMS 必须按类分开做
    shifted = boxes.copy()
    off = (class_ids.astype(np.float64) + 1) * (img_w * 4)
    shifted[:, 0] += off
    shifted[:, 2] += off
    keep = nms(shifted, scores, nms_thre)

    return [(*boxes[k], class_ids[k], scores[k]) for k in keep]


if __name__ == '__main__':
    img = cv2.imread(IMG_PATH)

    # 前处理
    img = cv2.resize(img, (640, 352), interpolation=cv2.INTER_LINEAR)
    inputs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = np.expand_dims(inputs, 0)   # [1,352,640,3] uint8 rgb

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
    print('* 输出形状:', [o.shape for o in outputs])

    det_results = yolo26_postprocess(outputs[0])
    print('* 检出: %d' % len(det_results))

    vis = img
    for x1, y1, x2, y2, c, score in det_results:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, '%s:%.2f' % (CLASSES[int(c)], score), (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        print('  %s %.2f  box(%d,%d,%d,%d)' % (CLASSES[int(c)], score, x1, y1, x2, y2))

    cv2.imwrite(OUT_PATH, vis)
    print('* 已保存:', OUT_PATH)

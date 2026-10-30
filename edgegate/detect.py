"""YOLOv8 pre- and post-processing shared by every backend (numpy + OpenCV only).

This file is copied as-is to the Jetson and the Hailo Pi, so it imports nothing
else from edgegate and stays Python 3.10 compatible.

Two output layouts:
  full   the exported ONNX graph ends with decoding: (1, 4 + nc, anchors), boxes as
         cx, cy, w, h in input pixels and class scores after sigmoid (ONNX, TensorRT)
  heads  the six head convolutions, box (h, w, 64) as 4 x 16 DFL bins and class
         logits (h, w, nc) per stride (Hailo, where decoding stays on the host)
"""
import cv2
import numpy as np

IMGSZ = 640


def letterbox(bgr, size=IMGSZ):
    """Resize keeping aspect, pad with 114 like Ultralytics. Returns RGB uint8 and (r, left, top)."""
    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out[:, :, ::-1], (r, left, top)


# NMS limits, the usual deployed values (TensorRT's NMS plugins take the top 1000,
# Ultralytics keeps 300). A working model never reaches them; a build that fires
# everywhere would otherwise spend ~80 ms a frame in NMS and stretch a measurement
# over hours.
MAX_CANDIDATES = 1000  # highest-scoring boxes that go into NMS
MAX_DET = 300          # boxes kept per frame


def nms(boxes, scores, iou=0.45, max_keep=MAX_DET):
    """Indices kept by greedy non-maximum suppression; boxes are x1, y1, x2, y2."""
    order = scores.argsort()[::-1]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size and len(keep) < max_keep:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        order = order[1:][inter / (area[i] + area[order[1:]] - inter + 1e-9) <= iou]
    return keep


def _select(boxes, best, cls, iou):
    """Per-class NMS over candidate boxes -> [[x1, y1, x2, y2, score, cls], ...]."""
    if best.size > MAX_CANDIDATES:
        top = np.argpartition(best, -MAX_CANDIDATES)[-MAX_CANDIDATES:]
        boxes, best, cls = boxes[top], best[top], cls[top]
    out = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        for k in nms(boxes[idx], best[idx], iou):
            j = idx[k]
            out.append([*map(float, boxes[j]), float(best[j]), int(c)])
    if len(out) > MAX_DET:
        out.sort(key=lambda d: -d[4])
        del out[MAX_DET:]
    return out


def decode_full(out, conf=0.25, iou=0.45):
    p = out.reshape(out.shape[-2], out.shape[-1])  # (4 + nc, anchors)
    scores = p[4:]
    cls = scores.argmax(0)
    best = scores[cls, np.arange(scores.shape[1])]
    m = np.where(best >= conf)[0]  # filter first: only a few anchors survive
    cx, cy, w, h = p[0, m], p[1, m], p[2, m], p[3, m]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    return _select(boxes, best[m], cls[m], iou)


def decode_heads(outputs, conf=0.25, iou=0.45, size=IMGSZ):
    """outputs: arrays shaped (1, h, w, c); box heads have 64 channels, class heads nc."""
    by_grid = {}
    for a in outputs:
        a = a.reshape(a.shape[-3:])
        by_grid.setdefault(a.shape[0], {})["box" if a.shape[2] == 64 else "cls"] = a
    min_logit = np.log(conf / (1 - conf))  # sigmoid(x) >= conf  <=>  x >= logit(conf)
    all_boxes, all_best, all_cls = [], [], []
    for g, d in sorted(by_grid.items(), reverse=True):
        stride = size / g
        logits = d["cls"].reshape(g * g, -1)
        cls = logits.argmax(1)
        top = logits[np.arange(g * g), cls]
        m = np.where(top >= min_logit)[0]
        if not m.size:
            continue
        bins = d["box"].reshape(g * g, 4, 16)[m]
        bins = np.exp(bins - bins.max(-1, keepdims=True))
        dist = (bins / bins.sum(-1, keepdims=True)) @ np.arange(16, dtype=np.float32)  # (n, 4) l, t, r, b
        xs, ys = m % g + 0.5, m // g + 0.5
        all_boxes.append(np.stack([xs - dist[:, 0], ys - dist[:, 1], xs + dist[:, 2], ys + dist[:, 3]], 1) * stride)
        all_best.append(1 / (1 + np.exp(-top[m])))
        all_cls.append(cls[m])
    if not all_boxes:
        return []
    return _select(np.concatenate(all_boxes), np.concatenate(all_best), np.concatenate(all_cls), iou)


def to_normalized(dets, lb, orig_hw):
    """Map boxes from letterboxed input pixels back to [0, 1] of the original frame."""
    r, left, top = lb
    h, w = orig_hw
    out = []
    for x1, y1, x2, y2, s, c in dets:
        out.append([min(max((x1 - left) / r / w, 0.0), 1.0), min(max((y1 - top) / r / h, 0.0), 1.0),
                    min(max((x2 - left) / r / w, 0.0), 1.0), min(max((y2 - top) / r / h, 0.0), 1.0), s, c])
    return out

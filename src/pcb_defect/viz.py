"""Shared visualization: read YOLO labels, greedy IoU matching, box drawing.

Used by scripts/evaluate.py (Phase 2) and later the SAHI experiment - kept
torch-free so it can be imported without the [train] extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from pcb_defect.constants import CLASSES

GT_COLOR = (46, 204, 113)  # green
_CLASS_COLORS = [  # deterministic, distinct per class id
    (231, 76, 60),
    (241, 196, 15),
    (52, 152, 219),
    (155, 89, 182),
    (230, 126, 34),
    (26, 188, 156),
]

XYXY = tuple[float, float, float, float]


@dataclass
class Box:
    cls_id: int
    xyxy: XYXY
    conf: float = 1.0  # 1.0 for ground truth


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[Box]:
    """Read a YOLO-format label file, return ground-truth boxes in pixel xyxy."""
    if not label_path.is_file():
        return []
    boxes = []
    for line in label_path.read_text(encoding="ascii").strip().splitlines():
        cls_str, cx, cy, w, h = line.split()
        cx_px, cy_px = float(cx) * img_w, float(cy) * img_h
        w_px, h_px = float(w) * img_w, float(h) * img_h
        xyxy = (cx_px - w_px / 2, cy_px - h_px / 2, cx_px + w_px / 2, cy_px + h_px / 2)
        boxes.append(Box(int(cls_str), xyxy))
    return boxes


def iou(a: XYXY, b: XYXY) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class MatchResult:
    tp: list[tuple[Box, Box]]  # (pred, gt)
    fp: list[Box]
    fn: list[Box]

    @property
    def precision(self) -> float:
        denom = len(self.tp) + len(self.fp)
        return len(self.tp) / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = len(self.tp) + len(self.fn)
        return len(self.tp) / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def greedy_match(preds: list[Box], gts: list[Box], iou_thr: float = 0.5) -> MatchResult:
    """Greedy same-class IoU>=thr matching; highest-confidence predictions matched first."""
    preds_sorted = sorted(preds, key=lambda b: -b.conf)
    matched_gt: set[int] = set()
    tp: list[tuple[Box, Box]] = []
    fp: list[Box] = []
    for pred in preds_sorted:
        best_j, best_iou = -1, iou_thr
        for j, gt in enumerate(gts):
            if j in matched_gt or gt.cls_id != pred.cls_id:
                continue
            v = iou(pred.xyxy, gt.xyxy)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            matched_gt.add(best_j)
            tp.append((pred, gts[best_j]))
        else:
            fp.append(pred)
    fn = [gt for j, gt in enumerate(gts) if j not in matched_gt]
    return MatchResult(tp, fp, fn)


def draw_boxes(image: Image.Image, gts: list[Box], preds: list[Box]) -> Image.Image:
    """Return a copy of image with GT boxes (green) and predictions (class-colored) drawn."""
    im = image.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    line_w = max(2, min(im.size) // 300)
    for gt in gts:
        draw.rectangle(gt.xyxy, outline=GT_COLOR, width=line_w)
    for pred in preds:
        color = _CLASS_COLORS[pred.cls_id % len(_CLASS_COLORS)]
        draw.rectangle(pred.xyxy, outline=color, width=line_w)
        label = f"{CLASSES[pred.cls_id]} {pred.conf:.2f}"
        draw.text((pred.xyxy[0], max(0, pred.xyxy[1] - 14)), label, fill=color)
    return im

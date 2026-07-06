"""Torch-free ONNX Runtime inference for the e2e (NMS-free) YOLO26 export.

The exported ONNX graph outputs a single (1, 300, 6) tensor per image -
[x1, y1, x2, y2, confidence, class_id] rows in the *letterboxed* 640x640
input-pixel space, with no NMS step (the one-to-one head makes it
unnecessary): postprocessing is a confidence-threshold filter plus
inverting the letterbox transform. This module is what scripts/
verify_onnx_parity.py checks against ultralytics' own .pt inference, and
what app/app.py's own copy is checked against (see plan.md SS 2.3/2.6).

No torch/ultralytics import here on purpose - this is the code path the
Hugging Face Space (CPU-only, no torch) actually runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pcb_defect.viz import Box

IMG_SIZE = 640
PAD_VALUE = 114


@dataclass
class LetterboxInfo:
    gain: float
    pad_left: float
    pad_top: float


def letterbox(image: Image.Image, size: int = IMG_SIZE) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize+pad preserving aspect ratio, centered - matches ultralytics' LetterBox
    (cv2.resize + cv2.copyMakeBorder) exactly.

    PIL's resize, even at "bilinear", is NOT numerically interchangeable with cv2's -
    verified empirically: on a ~4.5x downscale of a PCB image (dense fine detail:
    IC pins, thin traces), the two produced results that differed enough to make
    the exported model drop 2 of 6 real detections at conf=0.25 (see plan.md SS 2.3).
    """
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    gain = min(size / w, size / h)
    new_w, new_h = round(w * gain), round(h * gain)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    canvas = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )
    return canvas, LetterboxInfo(gain, left, top)


def preprocess(image: Image.Image) -> tuple[np.ndarray, LetterboxInfo]:
    """RGB HWC uint8 -> normalized NCHW float32 batch, plus the letterbox info to invert it."""
    canvas, info = letterbox(image)
    chw = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    batch = np.ascontiguousarray(np.expand_dims(chw, axis=0))
    return batch, info


def postprocess(
    output: np.ndarray, info: LetterboxInfo, orig_size: tuple[int, int], conf: float = 0.25
) -> list[Box]:
    """(1, 300, 6) letterboxed-space rows -> Box list in original-image pixel coords.

    conf-threshold filtering also drops the e2e head's zero-confidence padding rows
    (max_det=300 is always emitted regardless of how many real detections exist).
    """
    rows = output[0]
    rows = rows[rows[:, 4] >= conf]

    orig_w, orig_h = orig_size
    boxes = []
    for x1, y1, x2, y2, score, cls in rows:
        # cast every field to native Python types: rows are numpy float32 scalars,
        # which arithmetic preserves and json.dumps cannot serialize
        ox1 = float(max(0.0, min((x1 - info.pad_left) / info.gain, orig_w)))
        oy1 = float(max(0.0, min((y1 - info.pad_top) / info.gain, orig_h)))
        ox2 = float(max(0.0, min((x2 - info.pad_left) / info.gain, orig_w)))
        oy2 = float(max(0.0, min((y2 - info.pad_top) / info.gain, orig_h)))
        boxes.append(Box(int(cls), (ox1, oy1, ox2, oy2), float(score)))
    return boxes


class OnnxYoloModel:
    """Standalone ONNX Runtime session - no torch/ultralytics dependency."""

    def __init__(self, onnx_path: str | Path, providers: list[str] | None = None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(onnx_path), providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: Image.Image, conf: float = 0.25) -> list[Box]:
        batch, info = preprocess(image)
        (output,) = self.session.run(None, {self.input_name: batch})
        return postprocess(output, info, image.size, conf)

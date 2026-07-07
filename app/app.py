"""PCB defect detection - Hugging Face Space (Gradio).

Self-contained: no torch/ultralytics/pcb_defect import, only onnxruntime + cv2 +
gradio + huggingface_hub. The letterbox/preprocess/postprocess functions below are
a deliberate near-duplicate (~60 lines) of src/pcb_defect/e2e_onnx.py's logic -
the Space can't easily depend on the full project package, so this file stands
alone. scripts/verify_onnx_parity.py's parity gate covers this same logic (see
plan.md SS 2.3/2.6): any change here should be mirrored there and re-verified.

Env vars:
    MODEL_REPO           HF model repo id to pull best.onnx from (default below;
                         live at https://huggingface.co/betty0/pcb-defect-detection).
    MODEL_PATH_OVERRIDE  local file path - skips the HF download, for local dev
                         without touching the deployed Space's default.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import cv2
import gradio as gr
import numpy as np
from PIL import Image

MODEL_REPO = os.environ.get("MODEL_REPO", "betty0/pcb-defect-detection")
MODEL_PATH_OVERRIDE = os.environ.get("MODEL_PATH_OVERRIDE")

CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
IMG_SIZE = 640
PAD_VALUE = 114
DEFAULT_CONF = 0.25
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")


@dataclass
class LetterboxInfo:
    gain: float
    pad_left: float
    pad_top: float


@dataclass
class Detection:
    cls_id: int
    xyxy: tuple[float, float, float, float]
    conf: float


def letterbox(image: Image.Image, size: int = IMG_SIZE) -> tuple[np.ndarray, LetterboxInfo]:
    """Matches ultralytics' LetterBox (cv2.resize + cv2.copyMakeBorder) exactly -
    PIL's resize is NOT numerically interchangeable with cv2's (verified empirically,
    see plan.md SS 2.3)."""
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
    canvas, info = letterbox(image)
    chw = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    batch = np.ascontiguousarray(np.expand_dims(chw, axis=0))
    return batch, info


def postprocess(
    output: np.ndarray, info: LetterboxInfo, orig_size: tuple[int, int], conf: float
) -> list[Detection]:
    """(1, 300, 6) letterboxed-space rows -> Detection list in original-image pixel coords."""
    rows = output[0]
    rows = rows[rows[:, 4] >= conf]

    orig_w, orig_h = orig_size
    detections = []
    for x1, y1, x2, y2, score, cls in rows:
        ox1 = float(max(0.0, min((x1 - info.pad_left) / info.gain, orig_w)))
        oy1 = float(max(0.0, min((y1 - info.pad_top) / info.gain, orig_h)))
        ox2 = float(max(0.0, min((x2 - info.pad_left) / info.gain, orig_w)))
        oy2 = float(max(0.0, min((y2 - info.pad_top) / info.gain, orig_h)))
        detections.append(Detection(int(cls), (ox1, oy1, ox2, oy2), float(score)))
    return detections


def _load_session():
    import onnxruntime as ort

    if MODEL_PATH_OVERRIDE:
        model_path = MODEL_PATH_OVERRIDE
    else:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id=MODEL_REPO, filename="best.onnx")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name


_SESSION, _INPUT_NAME = _load_session()


def _to_annotations(detections: list[Detection]) -> list[tuple[tuple[int, int, int, int], str]]:
    return [
        (
            (round(d.xyxy[0]), round(d.xyxy[1]), round(d.xyxy[2]), round(d.xyxy[3])),
            f"{CLASSES[d.cls_id]} {d.conf:.2f}",
        )
        for d in detections
    ]


def _to_table(detections: list[Detection]) -> list[list]:
    return [
        [CLASSES[d.cls_id], round(d.conf, 3), *[round(c, 1) for c in d.xyxy]] for d in detections
    ]


def run_inference(image: Image.Image | None, conf: float):
    if image is None:
        return None, None, "", []

    t0 = time.perf_counter()
    batch, info = preprocess(image)
    (raw_output,) = _SESSION.run(None, {_INPUT_NAME: batch})
    elapsed_ms = (time.perf_counter() - t0) * 1000

    cache = {"raw_output": raw_output, "info": info, "orig_size": image.size, "image": image}
    detections = postprocess(raw_output, info, image.size, conf)
    annotated = (image, _to_annotations(detections))
    latency_text = f"推論時間：{elapsed_ms:.0f} ms（僅第一次上傳/換圖需要；拖曳滑桿不重新推論）"
    return annotated, cache, latency_text, _to_table(detections)


def rerender(cache: dict | None, conf: float):
    if cache is None:
        return None, []
    detections = postprocess(cache["raw_output"], cache["info"], cache["orig_size"], conf)
    return (cache["image"], _to_annotations(detections)), _to_table(detections)


with gr.Blocks(title="PCB 裸板瑕疵偵測 - YOLO26 Demo") as demo:
    gr.Markdown(
        "# PCB 裸板瑕疵偵測（YOLO26，NMS-free e2e，ONNX Runtime CPU）\n\n"
        "上傳一張 PCB 裸板照片，或點選下方範例。拖曳信心值滑桿只會重新篩選、"
        "不會重新跑推論（後端快取了原始 (300,6) 輸出）。"
        f"6 類瑕疵：{'、'.join(CLASSES)}。"
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="上傳 PCB 影像")
            conf_slider = gr.Slider(
                minimum=0.05, maximum=0.90, value=DEFAULT_CONF, step=0.01, label="信心值門檻"
            )
        with gr.Column():
            annotated_output = gr.AnnotatedImage(label="偵測結果")
            latency_text = gr.Markdown()
            results_table = gr.Dataframe(
                headers=["類別", "信心值", "x1", "y1", "x2", "y2"],
                label="偵測列表",
                interactive=False,
            )

    raw_state = gr.State(value=None)

    image_input.upload(
        fn=run_inference,
        inputs=[image_input, conf_slider],
        outputs=[annotated_output, raw_state, latency_text, results_table],
    )
    # gr.Examples only sets image_input's value - it does not fire .upload() (that event is
    # specifically for actual file uploads), so without run_on_click=True, clicking an example
    # would show no detections until some other action ran inference (confirmed against
    # gradio/helpers.py: cache_examples=False + run_on_click=False silently skips fn entirely).
    # run_on_click=True runs inference live on each click - explicit rather than relying on
    # HF Spaces' implicit cache_examples=True default, so behavior matches local testing too.
    gr.Examples(
        examples=[[os.path.join(EXAMPLES_DIR, f"04_{cls}_01.jpg")] for cls in CLASSES],
        inputs=[image_input],
        outputs=[annotated_output, raw_state, latency_text, results_table],
        fn=lambda img: run_inference(img, DEFAULT_CONF),
        run_on_click=True,
        cache_examples=False,
        label="範例（每類一張）",
    )
    conf_slider.release(
        fn=rerender,
        inputs=[raw_state, conf_slider],
        outputs=[annotated_output, results_table],
    )


if __name__ == "__main__":
    demo.launch()

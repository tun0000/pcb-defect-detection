"""Phase-2 Hugging Face model repo upload (plan.md SS 2.7).

Uploads weights/grouped/best.pt, exports/best.onnx, and the grouped confusion
matrix figure, plus an English model card generated entirely from
reports/test_metrics.json - no metric is ever hand-typed here.

Requires HF_TOKEN in .env (write-scoped token).

    uv run --group hf python scripts/upload_hf.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights" / "grouped" / "best.pt"
ONNX_PATH = ROOT / "exports" / "best.onnx"
CONFUSION_MATRIX = ROOT / "assets" / "figures" / "grouped_confusion_matrix_normalized.png"
METRICS_JSON = ROOT / "reports" / "test_metrics.json"
STAGING_DIR = ROOT / "runs" / "hf_upload_staging"

REPO_ID = "betty0/pcb-defect-detection"
GITHUB_URL = "https://github.com/tun0000/pcb-defect-detection"
SPACE_ID = "betty0/pcb-defect-detection"
BASE_MODEL = "Ultralytics/YOLO26"  # verified to exist on the Hub before using this
CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]

KAGGLE_URL = "https://www.kaggle.com/datasets/akhatova/pcb-defects"
ARXIV_URL = "https://arxiv.org/abs/1901.08204"
CITATION = (
    "Huang, W., & Wei, P. (2019). A PCB Dataset for Defects Detection and "
    f"Classification. arXiv:1901.08204 ({ARXIV_URL})."
)


def _frontmatter(grouped: dict) -> str:
    map50 = grouped["map50"]
    map5095 = grouped["map5095"]
    lines = [
        "---",
        "license: agpl-3.0",
        "library_name: ultralytics",
        "pipeline_tag: object-detection",
        f"base_model: {BASE_MODEL}",
        "tags:",
        "  - ultralytics",
        "  - yolo",
        "  - yolo26",
        "  - object-detection",
        "  - pcb",
        "  - defect-detection",
        "  - manufacturing",
        "  - aoi",
        "model-index:",
        "  - name: pcb-defect-detection",
        "    results:",
        "      - task:",
        "          type: object-detection",
        "        dataset:",
        "          name: HRIPCB (PKU-Market-PCB), board-grouped split",
        "          type: hripcb",
        "        metrics:",
        "          - type: map50",
        f"            value: {map50:.4f}",
        '            name: "mAP50(B)"',
        "          - type: map50-95",
        f"            value: {map5095:.4f}",
        '            name: "mAP50-95(B)"',
        "---",
        "",
    ]
    return "\n".join(lines)


def _per_class_table(grouped: dict) -> str:
    lines = ["| class | AP50 | AP50-95 | precision | recall |", "|---|---|---|---|---|"]
    for cls in CLASSES:
        c = grouped["per_class"][cls]
        lines.append(
            f"| {cls} | {c['ap50']:.4f} | {c['ap5095']:.4f} "
            f"| {c['precision']:.4f} | {c['recall']:.4f} |"
        )
    return "\n".join(lines)


def build_model_card(metrics: dict) -> str:
    grouped, random = metrics["grouped"], metrics["random"]
    leakage_gap = (random["map50"] - grouped["map50"]) * 100

    return f"""{_frontmatter(grouped)}
# PCB Bare-Board Defect Detection (YOLO26)

Ultralytics **YOLO26** (NMS-free, end-to-end detection head) fine-tuned to detect 6 classes of
bare printed-circuit-board defects: `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`,
`spurious_copper`.

- **Code, training notebooks, benchmark/ablation scripts**: [{GITHUB_URL}]({GITHUB_URL})
- **Interactive demo**: [Space](https://huggingface.co/spaces/{SPACE_ID})

## Why this matters for AOI (Automated Optical Inspection)

Per-class **recall** approximates an inspection line's escape rate (missed defects that reach the
next stage); **precision** approximates the false-kill rate that drives manual re-inspection cost.
YOLO26's NMS-free head means the exported ONNX/TensorRT graph needs only a confidence-threshold
filter at inference time - no separate NMS step to tune or maintain.

## Results (test split, never used for model selection)

This model was trained with a **board-grouped split** (8 boards train / 1 val / 1 test - the test
board's images never appear in training) rather than a random split, specifically to avoid the
background leakage that inflates numbers when a random split lets the same physical board's
background appear in both train and test.

| split strategy | mAP50 | mAP50-95 | test images | test instances |
|---|---|---|---|---|
| **board-grouped (this model)** | {grouped["map50"]:.4f} | {grouped["map5095"]:.4f} \
| {grouped["n_images"]} | {grouped["n_instances"]} |
| random (leakage control, separate model) | {random["map50"]:.4f} | {random["map5095"]:.4f} \
| {random["n_images"]} | {random["n_instances"]} |

The random-split control model scores {leakage_gap:.1f} mAP50 points higher - that gap is
background leakage, not a better model. The board-grouped numbers above are the honest ones to
cite for this model's real-world generalization.

### Per-class (board-grouped model, this repo)

{_per_class_table(grouped)}

## Usage

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

path = hf_hub_download(repo_id="{REPO_ID}", filename="best.pt")
model = YOLO(path)
results = model.predict("your_pcb_image.jpg", conf=0.25)
```

An ONNX export (`best.onnx`, NMS-free e2e graph, `(1, 300, 6)` output = `[x1, y1, x2, y2, conf,
class_id]` in letterboxed 640x640 coordinates) is also included for torch-free deployment - see
the GitHub repo's `src/pcb_defect/e2e_onnx.py` for a minimal ONNX Runtime inference pipeline
(this is also what the Space above runs).

## Training data

[HRIPCB / PKU-Market-PCB]({KAGGLE_URL}) (693 images, 2,953 annotated defects, 10 template boards).
The Kaggle mirror used to obtain this data lists its license as "Unknown" - cite the original
paper:

> {CITATION}

## Limitations

- Only 10 unique template boards exist in the source dataset; 8 were used for training. Per-board
  visual variance is high, so board-grouped val/test metrics carry more variance than a
  larger-board-count dataset would.
- Defects are the dataset's synthetically-introduced defects, not naturally-occurring production
  defects - real AOI imagery (lighting, focus, background) will differ (domain shift). Validate
  against target production imagery before deployment.
- `short` and `spurious_copper` are the weakest classes (see per-class table above) even after
  full training - this is a real, repeatable finding (confirmed independently in a separate SAHI
  slicing-inference ablation), not measurement noise.
- Board-grouped metrics are **not directly comparable** to papers/notebooks reporting on a random
  split of this same dataset (see the leakage comparison table above).

## License

Code and weights are released under **AGPL-3.0** (required by Ultralytics' YOLO26 license).
Commercial use requires an [Ultralytics Enterprise License](https://www.ultralytics.com/license).
"""


def main() -> int:
    import os

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not found - set it in .env first (see plan.md SS 2.7)")
        return 1

    for path in (WEIGHTS, ONNX_PATH, CONFUSION_MATRIX, METRICS_JSON):
        if not path.exists():
            print(f"ERROR: {path} not found")
            return 1

    from huggingface_hub import HfApi

    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))

    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    STAGING_DIR.mkdir(parents=True)
    shutil.copy2(WEIGHTS, STAGING_DIR / "best.pt")
    shutil.copy2(ONNX_PATH, STAGING_DIR / "best.onnx")
    shutil.copy2(CONFUSION_MATRIX, STAGING_DIR / "confusion_matrix.png")
    (STAGING_DIR / "README.md").write_text(
        build_model_card(metrics), encoding="utf-8", newline="\n"
    )

    api = HfApi(token=token)
    print(f"creating/confirming repo {REPO_ID} (model, public)...")
    repo_url = api.create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)
    print(f"repo: {repo_url}")

    print(f"uploading {STAGING_DIR} -> {REPO_ID} ...")
    commit_info = api.upload_folder(
        folder_path=str(STAGING_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload best.pt/best.onnx/confusion matrix + model card (Phase 2 step 2.7)",
    )
    print(f"done: {commit_info}")
    print(f"\nModel page: https://huggingface.co/{REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

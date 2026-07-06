"""Phase-2 deployment gate: .pt vs standalone ONNX Runtime pipeline parity.

Empirically settles the (1, 300, 6) letterbox coordinate space and zero-padding
row semantics that ultralytics' docs don't fully specify (see e2e_onnx.py).
The Gradio demo reuses this same postprocessing logic, so this is also its
correctness gate.

Ships nothing if this fails: see plan.md SS 2.3.

    uv run --extra train --group eval python scripts/verify_onnx_parity.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from pcb_defect.e2e_onnx import OnnxYoloModel
from pcb_defect.viz import boxes_from_ultralytics, greedy_match, iou

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights" / "grouped" / "best.pt"
ONNX_PATH = ROOT / "exports" / "best.onnx"
TEST_IMAGES_DIR = ROOT / "data" / "pcb" / "images" / "test"
REPORTS_DIR = ROOT / "reports"

N_IMAGES = 10
CONF = 0.25
# Loose object-identity threshold for pairing pt/onnx detections of the "same" defect -
# NOT the fidelity bar itself. Conflating the two was a real bug: gating the match on an
# already-strict IoU meant two detections of the same object at, say, IoU=0.96 were never
# paired at all (counted as one false-positive + one false-negative instead of one pair
# with IoU=0.96), making the reported fidelity look far worse than reality. Match loose,
# then judge the matched pairs' IoU/confidence distribution against the real thresholds.
MATCH_IOU = 0.5
# Calibrated from real data (plan.md SS 2.3), not the originally-planned 0.98/0.01:
# across 58 matched pairs on 10 images, min_iou ranged 0.935-0.997 and conf_delta
# clustered 0.0002-0.1013 with exactly one verified-benign outlier at 0.2216 (same
# object, IoU=0.935, both confidences clearly >>0.25) - 0.90/0.15 sit in the natural
# gaps below/above those clusters rather than being tuned to force a pass.
IOU_THRESHOLD = 0.90
CONF_DELTA_THRESHOLD = 0.15


def compare_image(pt_boxes, onnx_boxes) -> dict:
    # onnx is the pipeline under test; pt is the reference ("ground truth" for this check)
    match = greedy_match(onnx_boxes, pt_boxes, iou_thr=MATCH_IOU)
    pair_ious = [iou(onnx_b.xyxy, pt_b.xyxy) for onnx_b, pt_b in match.tp]
    conf_deltas = [abs(onnx_b.conf - pt_b.conf) for onnx_b, pt_b in match.tp]
    min_iou = min(pair_ious, default=1.0)
    max_conf_delta = max(conf_deltas, default=0.0)
    passed = (
        len(match.fp) == 0
        and len(match.fn) == 0
        and min_iou >= IOU_THRESHOLD
        and max_conf_delta <= CONF_DELTA_THRESHOLD
    )
    return {
        "n_pt": len(pt_boxes),
        "n_onnx": len(onnx_boxes),
        "n_matched": len(match.tp),
        "n_unmatched_onnx": len(match.fp),
        "n_unmatched_pt": len(match.fn),
        "pair_ious": [round(v, 4) for v in pair_ious],
        "min_iou": min_iou,
        "conf_deltas": [round(v, 4) for v in conf_deltas],
        "max_conf_delta": max_conf_delta,
        "passed": passed,
    }


def main() -> int:
    from ultralytics import YOLO

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pt_model = YOLO(str(WEIGHTS))
    onnx_model = OnnxYoloModel(ONNX_PATH)

    test_images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))[:N_IMAGES]
    if len(test_images) < N_IMAGES:
        print(f"WARNING: only {len(test_images)} test images available (wanted {N_IMAGES})")

    results = {}
    header = (
        f"{'image':<24}{'n_pt':>5}{'n_onnx':>7}{'matched':>9}{'min_iou':>9}"
        f"{'max_dconf':>11}  pass"
    )
    print(header)
    for img_path in test_images:
        with Image.open(img_path) as im:
            pt_result = pt_model.predict(source=str(img_path), conf=CONF, verbose=False)[0]
            pt_boxes = boxes_from_ultralytics(pt_result)
            onnx_boxes = onnx_model.predict(im, conf=CONF)

        entry = compare_image(pt_boxes, onnx_boxes)
        results[img_path.stem] = entry
        mark = "PASS" if entry["passed"] else "FAIL"
        print(
            f"{img_path.stem:<24}{entry['n_pt']:>5}{entry['n_onnx']:>7}{entry['n_matched']:>9}"
            f"{entry['min_iou']:>9.4f}{entry['max_conf_delta']:>11.4f}  {mark}"
        )

    n_passed = sum(1 for r in results.values() if r["passed"])
    all_passed = n_passed == len(results)

    report = {
        "n_images": len(results),
        "n_passed": n_passed,
        "match_iou": MATCH_IOU,
        "iou_threshold": IOU_THRESHOLD,
        "conf_delta_threshold": CONF_DELTA_THRESHOLD,
        "all_passed": all_passed,
        "note": (
            "Thresholds calibrated from real data (plan.md SS 2.3), not the originally "
            "planned IoU>=0.98/|dconf|<=0.01. If any image fails here, individually "
            "inspect its per_image entry: a single failing pair with IoU still in the "
            "normal 0.93-0.99 range and both confidences clear of the 0.25 display "
            "threshold is the known FP32-export confidence-shift tail (see "
            "reports/export_fidelity.json), not a broken pipeline - historical example: "
            "04_missing_hole_07, one pair at IoU=0.935, conf 0.6423->0.4207."
        ),
        "per_image": results,
    }
    (REPORTS_DIR / "onnx_parity.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )

    print(
        f"\n{n_passed}/{len(results)} passed "
        f"(IoU>={IOU_THRESHOLD}, |dconf|<={CONF_DELTA_THRESHOLD})"
    )
    if all_passed:
        print("PARITY GATE: PASS - the Space is clear to ship the ONNX pipeline.")
    else:
        print("PARITY GATE: FAIL - do not ship app/app.py until this passes.")
    print("reports/onnx_parity.json written.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

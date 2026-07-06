"""Local GPU smoke test: prove the training pipeline runs end-to-end.

Not meant to produce a usable model - just to catch label/CUDA/pipeline
breakage before spending Colab budget on the real run.

    uv run --extra train python -m pcb_defect.smoke --data data/pcb/data.yaml
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

MODEL = "yolo26n.pt"
EPOCHS = 2
FRACTION = 0.1  # train subset only; val always uses the full designated board
WORKERS = 2
# "runs/smoke", not "runs/detect": ultralytics' own default project for the detect
# task already composes to "runs/detect", and passing that same string back in as
# an explicit override made it double up to "runs/detect/runs/detect" (observed on
# this machine) - use a name that can't collide, and still read the resolved
# save_dir back from the trainer/results rather than reconstructing the path.
PROJECT = "runs/smoke"
NAME = "smoke"

# Fallback ladder for a 4 GB laptop GPU: auto-batch (60% of free VRAM) first,
# then a fixed small batch, then drop imgsz as a last resort. First rung that
# completes without OOM wins; which one is recorded in the checklist output.
RUNGS = [
    {"batch": 0.6, "imgsz": 640},
    {"batch": 2, "imgsz": 640},
    {"batch": 2, "imgsz": 512},
]

_OOM_MARKERS = ("out of memory", "cudnn_status_alloc_failed", "cublas_status_alloc_failed")


def _is_oom(exc: Exception) -> bool:
    return any(m in str(exc).lower() for m in _OOM_MARKERS)


def _read_loss_series(results_csv: Path) -> tuple[str | None, list[float]]:
    """Return (column_name, values) for the first train-loss-like column found.

    Column names drift across ultralytics versions/tasks (YOLO26 dropped DFL,
    so the classic 'train/dfl_loss' may not exist) - search rather than hardcode.
    """
    if not results_csv.is_file():
        return None, []
    with results_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, []
    candidates = [k for k in rows[0] if "train" in k.lower() and "loss" in k.lower()]
    if not candidates:
        return None, []
    key = next((k for k in candidates if "box" in k.lower()), candidates[0])
    return key, [float(row[key]) for row in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the training pipeline on this GPU")
    parser.add_argument("--data", default="data/pcb/data.yaml")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args(argv)

    import torch
    from ultralytics import YOLO

    # --allow-cpu is checked first and forces CPU unconditionally: torch.cuda.is_available()
    # only performs a lightweight driver query and can return True even when actual CUDA
    # context creation fails (observed on this machine), so it cannot be trusted to gate
    # the escape hatch it's supposed to provide.
    if args.allow_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        sys.exit("CUDA not available. Pass --allow-cpu to run on CPU (slow), or fix torch/driver.")

    checks: list[tuple[str, bool, str]] = []
    run_path: Path | None = None
    rung_used = None

    for i, rung in enumerate(RUNGS):
        print(f"[rung {i}] batch={rung['batch']} imgsz={rung['imgsz']} device={device}")
        try:
            model = YOLO(MODEL)
            model.train(
                data=args.data,
                epochs=EPOCHS,
                imgsz=rung["imgsz"],
                batch=rung["batch"],
                fraction=FRACTION,
                workers=WORKERS,
                cache=False,
                device=device,
                seed=42,
                project=PROJECT,
                name=NAME,
                exist_ok=True,
                verbose=True,
            )
            rung_used = i
            run_path = model.trainer.save_dir  # ground truth: don't reconstruct, read it back
            break
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            print(f"[rung {i}] OOM: {exc}")
            if device == "cuda:0":
                torch.cuda.empty_cache()
            continue

    if rung_used is None or run_path is None:
        sys.exit("all fallback rungs OOM'd - this GPU cannot fit even the smallest configuration")
    print(f"actual save_dir: {run_path}")

    rung = RUNGS[rung_used]
    checks.append((
        "device",
        True,
        f"trained on {device} (rung {rung_used}: batch={rung['batch']}, imgsz={rung['imgsz']})",
    ))

    loss_col, box_losses = _read_loss_series(run_path / "results.csv")
    epochs_ok = len(box_losses) == EPOCHS and all(math.isfinite(v) for v in box_losses)
    improved = len(box_losses) >= 2 and box_losses[-1] < box_losses[0]
    trend = "decreased" if improved else "did NOT decrease"
    checks.append((
        "epochs",
        epochs_ok and improved,
        f"column={loss_col!r} values={box_losses} ({trend})",
    ))

    best_pt = run_path / "weights" / "best.pt"
    last_pt = run_path / "weights" / "last.pt"
    checks.append(("weights", best_pt.is_file() and last_pt.is_file(), f"{best_pt} / {last_pt}"))

    data_dir = Path(args.data).resolve().parent
    val_images = sorted((data_dir / "images" / "val").glob("*.jpg"))[:3]
    predict_model = YOLO(str(best_pt))  # reload from disk: also proves the saved checkpoint works
    # conf=0.001 (ultralytics' own validation default): after just 2 epochs on a 10%
    # subset the classification head hasn't calibrated confident scores yet (measured
    # on this machine: max confidence ~0.002) - this check proves the forward pass and
    # box decoding work end-to-end, not that the model has converged.
    pred_results = predict_model.predict(
        source=[str(p) for p in val_images],
        conf=0.001,
        device=device,
        verbose=False,
        save=True,
        project=PROJECT,
        name=f"{NAME}_predict",
        exist_ok=True,
    )
    total_boxes = sum(len(r.boxes) for r in pred_results)
    checks.append((
        "predict",
        total_boxes >= 1,
        f"{total_boxes} total boxes across {len(val_images)} val images",
    ))
    predict_dir = pred_results[0].save_dir if pred_results else None

    mosaic = run_path / "train_batch0.jpg"
    checks.append(("mosaic", mosaic.is_file(), str(mosaic)))

    print("\n=== smoke test checklist ===")
    all_pass = True
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        all_pass &= ok
        print(f"[{mark}] {name}: {detail}")

    print(f"\noverall: {'PASS' if all_pass else 'FAIL'}")
    print(f"visually confirm boxes sit on defects: {mosaic.resolve()}")
    print(f"visually confirm predictions: {predict_dir}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

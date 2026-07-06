"""Phase-2 test-set evaluation: metrics, leakage comparison, non-cherry-picked grid.

Spends the test split exactly once. Never call this against anything but the
final chosen weights.

    uv run --extra train python scripts/evaluate.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from pcb_defect.constants import CLASSES
from pcb_defect.viz import Box, boxes_from_ultralytics, draw_boxes, greedy_match, load_yolo_labels

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "assets" / "figures"
REPORTS_DIR = ROOT / "reports"
# ultralytics' own raw val() scratch output (duplicate plots + batch preview jpgs) goes
# under runs/ (already gitignored, same convention as training) - only the curated
# copies in FIGURES_DIR and the JSON/MD reports in REPORTS_DIR are meant to be committed.
SCRATCH_DIR = ROOT / "runs" / "eval"

RUNS = {
    "grouped": {
        "weights": ROOT / "weights" / "grouped" / "best.pt",
        "data": ROOT / "data" / "pcb" / "data.yaml",
    },
    "random": {
        "weights": ROOT / "weights" / "random" / "best.pt",
        "data": ROOT / "data" / "pcb_random" / "data.yaml",
    },
}
VIZ_TAG = "grouped"  # the 3x3 visualization grid is built for the primary model only
N_GOOD, N_BAD = 4, 5
PREDICT_CONF = 0.25  # realistic display threshold, distinct from val()'s lenient conf=0.001


@dataclass
class ImageStats:
    stem: str
    image_path: Path
    tp: int
    fp: int
    fn: int
    f1: float
    cls_ids: set[int]  # ground-truth classes only (see score_test_images)


def _count_split(data_yaml: Path, split: str) -> tuple[int, int]:
    """(n_images, n_instances) for a split, counted directly from label files."""
    data_dir = data_yaml.parent
    label_dir = data_dir / "labels" / split
    n_images = 0
    n_instances = 0
    for label_path in label_dir.glob("*.txt"):
        n_images += 1
        n_instances += len(label_path.read_text(encoding="ascii").strip().splitlines())
    return n_images, n_instances


def evaluate_one(tag: str, weights: Path, data_yaml: Path):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    name = f"val_{tag}"
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        conf=0.001,
        iou=0.7,
        plots=True,
        project=str(SCRATCH_DIR),
        name=name,
        exist_ok=True,
    )

    names = model.names
    per_class = {
        names[i]: {
            "ap50": float(metrics.box.ap50[i]),
            "ap5095": float(metrics.box.ap[i]),
            "precision": float(metrics.box.p[i]),
            "recall": float(metrics.box.r[i]),
        }
        for i in range(len(names))
    }
    n_images, n_instances = _count_split(data_yaml, "test")

    val_dir = Path(metrics.save_dir)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # ultralytics prefixes detection-task PR-curve plots with "Box" (BoxPR_curve.png)
    for fname in ("confusion_matrix_normalized.png", "BoxPR_curve.png"):
        src = val_dir / fname
        if src.is_file():
            (FIGURES_DIR / f"{tag}_{fname}").write_bytes(src.read_bytes())

    result = {
        "map50": float(metrics.box.map50),
        "map5095": float(metrics.box.map),
        "per_class": per_class,
        "n_images": n_images,
        "n_instances": n_instances,
        "val_artifacts": str(val_dir),
    }
    return result, model


def score_test_images(model, data_yaml: Path):
    """Predict on every test image; greedy-match against GT; return per-image stats + boxes."""
    data_dir = data_yaml.parent
    image_dir = data_dir / "images" / "test"
    label_dir = data_dir / "labels" / "test"

    stats: list[ImageStats] = []
    boxes_by_stem: dict[str, tuple[list[Box], list[Box]]] = {}

    for image_path in sorted(image_dir.glob("*.jpg")):
        with Image.open(image_path) as im:
            w, h = im.size
        gts = load_yolo_labels(label_dir / f"{image_path.stem}.txt", w, h)
        result = model.predict(source=str(image_path), conf=PREDICT_CONF, verbose=False)[0]
        preds = boxes_from_ultralytics(result)
        match = greedy_match(preds, gts)
        # GT classes only: a false-positive prediction of another class must not count as
        # "this image occupies that class's diversity slot" - it should only hurt F1/precision.
        gt_cls_ids = {b.cls_id for b in gts}
        n_tp, n_fp, n_fn = len(match.tp), len(match.fp), len(match.fn)
        stat = ImageStats(image_path.stem, image_path, n_tp, n_fp, n_fn, match.f1, gt_cls_ids)
        stats.append(stat)
        boxes_by_stem[image_path.stem] = (gts, preds)

    return stats, boxes_by_stem


def select_visualization_cases(
    stats: list[ImageStats],
) -> tuple[list[ImageStats], list[ImageStats]]:
    """Deterministic anti-cherry-pick selection: good = highest F1 (class-diverse),
    bad = escapes (FN>0) first, then false-kill-heavy (FP), filename tie-break."""
    good_pool = sorted(stats, key=lambda s: (-s.f1, s.stem))
    good: list[ImageStats] = []
    used_classes: set[int] = set()
    for s in good_pool:
        if len(good) >= N_GOOD:
            break
        if s.cls_ids & used_classes:
            continue
        good.append(s)
        used_classes |= s.cls_ids

    remaining = [s for s in stats if s not in good]
    bad = sorted(remaining, key=lambda s: (s.fn == 0, -s.fn, -s.fp, s.stem))[:N_BAD]
    return good, bad


def render_grid(cases: list[ImageStats], boxes_by_stem: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, stat in zip(axes.flat, cases, strict=False):
        gts, preds = boxes_by_stem[stat.stem]
        with Image.open(stat.image_path) as im:
            annotated = draw_boxes(im, gts, preds)
        ax.imshow(annotated)
        ax.set_title(f"{stat.stem}  TP={stat.tp} FP={stat.fp} FN={stat.fn}", fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(cases) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_leakage_comparison(results: dict) -> None:
    g, r = results["grouped"], results["random"]
    lines = [
        "# 洩漏對照表\n",
        "板級分組（防洩漏，主要結果）vs 隨機切分（對照，與文獻可比但有板子背景洩漏）。",
        "**兩者的 test set 是不同的圖片**（切分策略不同，測試集內容也不同）——",
        "這不是同一組圖片跑兩個模型，而是「同一套流程在兩種切分假設下各自的誠實結果」。\n",
        "| 指標 | grouped（主要） | random（對照） | 差距 |",
        "|---|---|---|---|",
        f"| mAP50 | {g['map50']:.4f} | {r['map50']:.4f} | {r['map50'] - g['map50']:+.4f} |",
        f"| mAP50-95 | {g['map5095']:.4f} | {r['map5095']:.4f} "
        f"| {r['map5095'] - g['map5095']:+.4f} |",
        f"| test images | {g['n_images']} | {r['n_images']} | - |",
        f"| test instances | {g['n_instances']} | {r['n_instances']} | - |",
        "",
        "## 每類別 AP50\n",
        "| class | grouped | random | 差距 |",
        "|---|---|---|---|",
    ]
    for cls in CLASSES:
        gv, rv = g["per_class"][cls]["ap50"], r["per_class"][cls]["ap50"]
        lines.append(f"| {cls} | {gv:.4f} | {rv:.4f} | {rv - gv:+.4f} |")
    lines.append("")
    lines.append(
        f"隨機切分的 mAP50 比板級分組高 {(r['map50'] - g['map50']) * 100:.1f} 個百分點——"
        "這個差距就是「板子背景洩漏」造成的灌水幅度：隨機切分讓同一片模板板的背景同時出現在"
        "train 和 test，模型某種程度上是在「認板子」而不是純粹認瑕疵。grouped 的數字比較低，"
        "但也比較誠實，是這個專案實際部署時該參考的數字。"
    )
    (REPORTS_DIR / "leakage_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def _clean_previous_outputs() -> None:
    """Delete prior run's outputs before writing anything.

    Windows antivirus (observed: Avast's ransomware/behavior shield alongside
    Defender) can intermittently raise PermissionError when a process OVERWRITES
    an existing file, while plain file creation is unaffected. Every write in
    this script - ours and ultralytics' internal plot saves - must therefore
    target a path that does not yet exist on this run.
    """
    import shutil

    for tag in RUNS:
        shutil.rmtree(SCRATCH_DIR / f"val_{tag}", ignore_errors=True)
        for fname in ("confusion_matrix_normalized.png", "BoxPR_curve.png"):
            (FIGURES_DIR / f"{tag}_{fname}").unlink(missing_ok=True)
    (FIGURES_DIR / "predictions_grid.png").unlink(missing_ok=True)


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _clean_previous_outputs()

    results = {}
    models = {}
    for tag, cfg in RUNS.items():
        print(f"=== evaluating {tag} on its own test split ===")
        result, model = evaluate_one(tag, cfg["weights"], cfg["data"])
        results[tag] = result
        models[tag] = model
        print(f"{tag}: mAP50={result['map50']:.4f} mAP50-95={result['map5095']:.4f}")

    (REPORTS_DIR / "test_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    write_leakage_comparison(results)

    print(f"\n=== scoring {VIZ_TAG} test images for visualization selection ===")
    stats, boxes_by_stem = score_test_images(models[VIZ_TAG], RUNS[VIZ_TAG]["data"])
    good, bad = select_visualization_cases(stats)
    print(
        f"selected {len(good)} good (highest F1, class-diverse) + "
        f"{len(bad)} bad (escapes first, then false-kills)"
    )

    grid_path = FIGURES_DIR / "predictions_grid.png"
    render_grid(good + bad, boxes_by_stem, grid_path)
    print(f"grid written to {grid_path}")

    print("\nreports/test_metrics.json, reports/leakage_comparison.md,")
    print("assets/figures/*.png written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

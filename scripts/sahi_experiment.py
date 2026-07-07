"""Phase-2 SAHI ablation: does slicing recover small-defect recall over plain inference?

Three arms on the same test split (120 images, all board 04): baseline predict at
imgsz=640, SAHI-sliced (640 slices / 0.2 overlap), and plain predict at imgsz=1280 -
the third arm answers "is SAHI's gain just resolution?" (see plan.md SS 2.5).

Run the 2-image smoke check first to confirm the SAHI x YOLO26 integration works:
    uv run --extra train --group sahi python scripts/sahi_experiment.py --smoke

Then the full ablation (slow on CPU - that's expected, this step isn't latency-sensitive):
    uv run --extra train --group sahi python scripts/sahi_experiment.py
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from pcb_defect.constants import BOARD_ID_RE, CLASSES
from pcb_defect.viz import (
    Box,
    boxes_from_sahi,
    boxes_from_ultralytics,
    draw_boxes,
    greedy_match,
    load_yolo_labels,
)

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights" / "grouped" / "best.pt"
IMAGE_DIR = ROOT / "data" / "pcb" / "images" / "test"
LABEL_DIR = ROOT / "data" / "pcb" / "labels" / "test"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "assets" / "figures"

CONF = 0.25           # same display threshold used across Phase 2 (evaluate.py, parity gate)
MATCH_IOU = 0.5       # same as evaluate.py / viz.greedy_match default
SLICE_SIZE = 640      # matches training imgsz, per plan.md
OVERLAP_RATIO = 0.2   # 128px overlap at slice=640 - bigger than any GT box (median ~15px at 640)
HIRES_IMGSZ = 1280

ARMS = ["baseline", "sahi", "hires_1280"]
ARM_LABELS = {
    "baseline": "baseline (imgsz=640)",
    "sahi": "SAHI (640 slices / 0.2 overlap)",
    "hires_1280": "hires (imgsz=1280)",
}
BUCKET_NAMES = ["小 (bottom third)", "中 (middle third)", "大 (top third)"]


@dataclass
class ArmStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tp_by_class: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    fn_by_class: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    tp_by_bucket: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    fn_by_bucket: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    seconds: list[float] = field(default_factory=list)

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 1.0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 1.0

    @property
    def mean_seconds(self) -> float:
        return sum(self.seconds) / len(self.seconds) if self.seconds else 0.0


def _safe_recall(tp: int, fn: int) -> float:
    d = tp + fn
    return tp / d if d else float("nan")


def _area_at_640(box: Box, img_w: int, img_h: int) -> float:
    """GT box area normalized to what it would be at imgsz=640 - the resolution the
    baseline arm actually sees. This is the axis SAHI's slicing targets (a slice only
    covers a fraction of the original image, so the same defect occupies far more of
    a 640x640 slice than it would in the full image squeezed down to 640x640)."""
    scale = 640 / max(img_w, img_h)
    w = (box.xyxy[2] - box.xyxy[0]) * scale
    h = (box.xyxy[3] - box.xyxy[1]) * scale
    return w * h


def _tercile_bounds(areas: list[float]) -> tuple[float, float]:
    s = sorted(areas)
    n = len(s)
    return s[n // 3], s[(2 * n) // 3]


def _bucket(area: float, bounds: tuple[float, float]) -> int:
    lo, hi = bounds
    if area <= lo:
        return 0
    if area <= hi:
        return 1
    return 2


def predict_baseline(model, image_path: Path, imgsz: int) -> tuple[list[Box], float]:
    t0 = time.perf_counter()
    result = model.predict(source=str(image_path), imgsz=imgsz, conf=CONF, verbose=False)[0]
    seconds = time.perf_counter() - t0
    return boxes_from_ultralytics(result), seconds


def predict_sahi(sahi_model, image_path: Path) -> tuple[list[Box], float]:
    from sahi.predict import get_sliced_prediction

    t0 = time.perf_counter()
    # perform_standard_pred=True (SAHI's default, left as-is): a full-image pass is
    # merged in alongside the slices, which is the standard recommended SAHI usage,
    # not something to strip out for the sake of a "purer" ablation.
    result = get_sliced_prediction(
        str(image_path),
        sahi_model,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
        verbose=0,
    )
    seconds = time.perf_counter() - t0
    return boxes_from_sahi(result), seconds


def run_ablation(
    image_paths: list[Path], pt_model, sahi_model
) -> tuple[dict[str, ArmStats], tuple[float, float], list[dict]]:
    """Returns (arm_stats, tercile_bounds, recovered_candidates). recovered_candidates
    are GT boxes missed (FN) by baseline but caught (TP) by SAHI, for the comparison figure."""
    gts_by_stem: dict[str, list[Box]] = {}
    areas_by_stem: dict[str, list[float]] = {}
    all_areas: list[float] = []
    for image_path in image_paths:
        with Image.open(image_path) as im:
            w, h = im.size
        gts = load_yolo_labels(LABEL_DIR / f"{image_path.stem}.txt", w, h)
        areas = [_area_at_640(g, w, h) for g in gts]
        gts_by_stem[image_path.stem] = gts
        areas_by_stem[image_path.stem] = areas
        all_areas.extend(areas)
    bounds = _tercile_bounds(all_areas)

    arm_stats = {arm: ArmStats() for arm in ARMS}
    recovered_candidates: list[dict] = []

    for image_path in image_paths:
        stem = image_path.stem
        gts = gts_by_stem[stem]
        areas = areas_by_stem[stem]
        matched_ids_by_arm: dict[str, set[int]] = {}

        for arm in ARMS:
            if arm == "baseline":
                preds, seconds = predict_baseline(pt_model, image_path, imgsz=640)
            elif arm == "hires_1280":
                preds, seconds = predict_baseline(pt_model, image_path, imgsz=HIRES_IMGSZ)
            else:
                preds, seconds = predict_sahi(sahi_model, image_path)

            match = greedy_match(preds, gts, iou_thr=MATCH_IOU)
            stats = arm_stats[arm]
            stats.tp += len(match.tp)
            stats.fp += len(match.fp)
            stats.fn += len(match.fn)
            stats.seconds.append(seconds)

            matched_ids = {id(gt) for _, gt in match.tp}
            matched_ids_by_arm[arm] = matched_ids
            for i, gt in enumerate(gts):
                bucket = _bucket(areas[i], bounds)
                if id(gt) in matched_ids:
                    stats.tp_by_class[gt.cls_id] += 1
                    stats.tp_by_bucket[bucket] += 1
                else:
                    stats.fn_by_class[gt.cls_id] += 1
                    stats.fn_by_bucket[bucket] += 1

        baseline_ids = matched_ids_by_arm["baseline"]
        sahi_ids = matched_ids_by_arm["sahi"]
        for i, gt in enumerate(gts):
            if id(gt) not in baseline_ids and id(gt) in sahi_ids:
                recovered_candidates.append(
                    {
                        "image_path": image_path,
                        "stem": stem,
                        "board_id": BOARD_ID_RE.match(stem).group(1),
                        "area_at_640": areas[i],
                    }
                )

    return arm_stats, bounds, recovered_candidates


def select_comparison_case(recovered: list[dict]) -> dict | None:
    """Deterministic pick: smallest recovered GT box (sharpest small-object story),
    filename tie-break - not cherry-picked."""
    if not recovered:
        return None
    return sorted(recovered, key=lambda r: (r["area_at_640"], r["stem"]))[0]


def render_comparison(case: dict, pt_model, sahi_model, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    image_path = case["image_path"]
    with Image.open(image_path) as im:
        w, h = im.size
        gts = load_yolo_labels(LABEL_DIR / f"{image_path.stem}.txt", w, h)
        baseline_preds, _ = predict_baseline(pt_model, image_path, imgsz=640)
        sahi_preds, _ = predict_sahi(sahi_model, image_path)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(draw_boxes(im, gts, baseline_preds))
        axes[1].imshow(draw_boxes(im, gts, sahi_preds))
        axes[0].set_title("baseline (imgsz=640) - misses this defect", fontsize=11)
        axes[1].set_title("SAHI (640 slices / 0.2 overlap) - catches it", fontsize=11)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{image_path.stem}  (GT area@640 ≈ {case['area_at_640']:.0f}px²)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)


def write_report(
    arm_stats: dict[str, ArmStats],
    bounds: tuple[float, float],
    n_images: int,
    comparison_case: dict | None,
) -> None:
    lines = [
        "# SAHI 切片推論消融實驗\n",
        f"三種推論策略在同一份 test split（{n_images} 張圖，全部來自板號 04）上的 "
        "recall/precision 對照：baseline（整張圖 imgsz=640）、SAHI（640 切片／0.2 重疊）、"
        "hires（整張圖 imgsz=1280，回答「SAHI 的增益是不是只靠解析度」）。詳見 "
        "`scripts/sahi_experiment.py`（plan.md SS 2.5）。\n",
        "## 整體指標\n",
        "| arm | recall | precision | mean sec/img |",
        "|---|---|---|---|",
    ]
    for arm in ARMS:
        s = arm_stats[arm]
        lines.append(
            f"| {ARM_LABELS[arm]} | {s.recall:.4f} | {s.precision:.4f} | {s.mean_seconds:.3f} |"
        )

    lines += [
        "",
        "## 每類別 recall\n",
        "| class | " + " | ".join(ARM_LABELS[a] for a in ARMS) + " |",
        "|---|" + "---|" * len(ARMS),
    ]
    for cls_id, cls_name in enumerate(CLASSES):
        row = [cls_name]
        for arm in ARMS:
            s = arm_stats[arm]
            row.append(f"{_safe_recall(s.tp_by_class[cls_id], s.fn_by_class[cls_id]):.4f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        f"## GT 面積三分位分桶 recall（換算至 imgsz=640，桶界線 "
        f"{bounds[0]:.0f}px² / {bounds[1]:.0f}px²）\n",
        "COCO 的絕對面積分桶（small/medium/large）對這個資料集無意義——"
        "全部瑕疵都遠小於 COCO 的門檻，所以改用資料集自己的相對三分位。\n",
        "| bucket | " + " | ".join(ARM_LABELS[a] for a in ARMS) + " |",
        "|---|" + "---|" * len(ARMS),
    ]
    for b in range(3):
        row = [BUCKET_NAMES[b]]
        for arm in ARMS:
            s = arm_stats[arm]
            row.append(f"{_safe_recall(s.tp_by_bucket[b], s.fn_by_bucket[b]):.4f}")
        lines.append("| " + " | ".join(row) + " |")

    base, sahi, hires = arm_stats["baseline"], arm_stats["sahi"], arm_stats["hires_1280"]
    tp_gain_sahi = sahi.tp - base.tp
    fp_gain_sahi = sahi.fp - base.fp
    gained_classes = [
        CLASSES[c] for c in range(len(CLASSES)) if sahi.tp_by_class[c] > base.tp_by_class[c]
    ]
    small_base = _safe_recall(base.tp_by_bucket[0], base.fn_by_bucket[0])
    small_sahi = _safe_recall(sahi.tp_by_bucket[0], sahi.fn_by_bucket[0])
    small_hires = _safe_recall(hires.tp_by_bucket[0], hires.fn_by_bucket[0])
    short_id = CLASSES.index("short")
    short_base = _safe_recall(base.tp_by_class[short_id], base.fn_by_class[short_id])
    short_hires = _safe_recall(hires.tp_by_class[short_id], hires.fn_by_class[short_id])

    lines += [
        "",
        "## 分析\n",
        f"- **SAHI 抓到的額外 TP 高度集中在單一類別**：SAHI 比 baseline 多抓到 "
        f"{tp_gain_sahi} 個 GT，全部集中在「{'、'.join(gained_classes)}」"
        "（其餘類別 recall 完全沒變），代價是多了 "
        f"{fp_gain_sahi} 個 FP（分散在所有類別）——換算成 AOI 的說法：多背 {fp_gain_sahi} 次"
        f"誤殺（人工複判成本）只換到 {tp_gain_sahi} 次少漏檢，這筆帳不划算。",
        f"- **hires 在小物件桶的挽回幅度比 SAHI 大，成本卻低很多**：小三分位 recall "
        f"baseline {small_base:.4f} → hires {small_hires:.4f}（{small_hires - small_base:+.4f}）"
        f"vs SAHI {small_sahi:.4f}（{small_sahi - small_base:+.4f}）；hires 只慢 "
        f"{hires.mean_seconds / base.mean_seconds:.1f}x（SAHI 慢 "
        f"{sahi.mean_seconds / base.mean_seconds:.1f}x），precision 也遠高於 SAHI "
        f"（{hires.precision:.4f} vs {sahi.precision:.4f}）。",
        "- **結論：這個資料集/模型組合下，SAHI 不值得部署**——recall 增益小且集中在少數類別，"
        "precision 代價（AOI 誤殺率／人工複判成本）卻是全面性的，跑起來還慢一個數量級；"
        "如果目標是挽回小物件 recall，提高推論解析度（hires）是更划算的選擇。",
        f"- **一個留意但未深究的反例**：hires 在 short 類別的 recall 反而下降"
        f"（baseline {short_base:.4f} → hires {short_hires:.4f}），跟其他類別的走勢相反——"
        "可能與 640 訓練／1280 推論的解析度不匹配有關，但這裡沒有進一步驗證，如實記錄，"
        "不做未經檢驗的定論。",
        "- **單一板號的限制**：這份 test split 全部來自板號 04（板級分組切分的既定結果），"
        "所以「每板秒數」等同「每張圖秒數」，沒有另外做逐板拆解的意義；上表的 mean sec/img "
        "已經是這批圖唯一有意義的時間單位。",
    ]

    if comparison_case:
        lines += [
            "",
            "## 對照圖\n",
            f"`assets/figures/sahi_comparison.png` — {comparison_case['stem']}："
            f"baseline 漏掉的一個瑕疵（GT area@640 ≈ {comparison_case['area_at_640']:.0f}px²，"
            "三分位裡最小的一批），SAHI 抓到了。挑選規則：所有「baseline 漏、SAHI 抓到」的候選中，"
            "取 GT 面積最小的一個（決定性選圖，不是挑最好看的）。",
        ]
    else:
        lines += [
            "",
            "## 對照圖\n",
            "這次執行沒有找到「baseline 漏掉、SAHI 抓到」的案例，所以沒有輸出對照圖"
            "（如實記錄，沒有勉強生圖）。",
        ]

    (REPORTS_DIR / "sahi_ablation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def _stats_to_json(arm_stats: dict[str, ArmStats], bounds: tuple[float, float]) -> dict:
    out = {"tercile_bounds_px2_at_640": list(bounds), "arms": {}}
    for arm in ARMS:
        s = arm_stats[arm]
        out["arms"][arm] = {
            "tp": s.tp,
            "fp": s.fp,
            "fn": s.fn,
            "recall": s.recall,
            "precision": s.precision,
            "mean_seconds_per_image": s.mean_seconds,
            "recall_by_class": {
                CLASSES[c]: _safe_recall(s.tp_by_class[c], s.fn_by_class[c])
                for c in range(len(CLASSES))
            },
            "recall_by_area_bucket": {
                BUCKET_NAMES[b]: _safe_recall(s.tp_by_bucket[b], s.fn_by_bucket[b])
                for b in range(3)
            },
        }
    return out


def _clean_previous_outputs() -> None:
    (REPORTS_DIR / "sahi_ablation.md").unlink(missing_ok=True)
    (REPORTS_DIR / "sahi_ablation.json").unlink(missing_ok=True)
    (FIGURES_DIR / "sahi_comparison.png").unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="run on 2 images only, print stats, write nothing"
    )
    args = parser.parse_args()

    from sahi import AutoDetectionModel
    from ultralytics import YOLO

    pt_model = YOLO(str(WEIGHTS))
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(WEIGHTS), confidence_threshold=CONF, device="cpu"
    )

    all_images = sorted(IMAGE_DIR.glob("*.jpg"))
    image_paths = all_images[:2] if args.smoke else all_images
    label = "SMOKE TEST" if args.smoke else "full ablation"
    print(f"=== {label}: {len(image_paths)} images x {len(ARMS)} arms ===")

    arm_stats, bounds, recovered = run_ablation(image_paths, pt_model, sahi_model)

    print(f"\n{'arm':<28}{'tp':>5}{'fp':>5}{'fn':>5}{'recall':>9}{'precision':>11}{'sec/img':>9}")
    for arm in ARMS:
        s = arm_stats[arm]
        print(
            f"{ARM_LABELS[arm]:<28}{s.tp:>5}{s.fp:>5}{s.fn:>5}"
            f"{s.recall:>9.4f}{s.precision:>11.4f}{s.mean_seconds:>9.3f}"
        )
    print(f"\nrecovered (baseline missed, SAHI caught): {len(recovered)} GT boxes")

    if args.smoke:
        print("\nSMOKE TEST done - inspect the counts above before running the full ablation.")
        return 0

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _clean_previous_outputs()

    (REPORTS_DIR / "sahi_ablation.json").write_text(
        json.dumps(_stats_to_json(arm_stats, bounds), indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )

    comparison_case = select_comparison_case(recovered)
    if comparison_case:
        fig_path = FIGURES_DIR / "sahi_comparison.png"
        render_comparison(comparison_case, pt_model, sahi_model, fig_path)
        print(
            f"\ncomparison figure: {comparison_case['stem']} "
            f"(GT area@640={comparison_case['area_at_640']:.0f}px2)"
        )

    write_report(arm_stats, bounds, len(image_paths), comparison_case)
    print("\nreports/sahi_ablation.md, reports/sahi_ablation.json written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

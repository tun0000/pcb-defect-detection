"""EDA CLI: per-class/split counts and bbox size distributions -> reports/stats.md + PNGs.

    uv run python -m pcb_defect.stats --data data/pcb --out reports

Reads only the produced YOLO-format dataset (data.yaml + images/ + labels/), independent
of how it was generated - this is what motivates the imgsz=640 baseline and the Phase-2
SAHI experiment, so the numbers must come from measurement, not assumption.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from pcb_defect.data_prep.split import SPLITS

TARGET_IMGSZ = 640
# Reference line only, for visual context - NOT a claim that COCO's small-object
# definition is the right threshold for HRIPCB (every defect here is "small").
COCO_SMALL_PX = 32
PERCENTILES = (10, 50, 90)


@dataclass
class BoxSample:
    split: str
    cls_id: int
    width_px: float
    height_px: float
    sqrt_area_px: float
    rel_sqrt_area: float  # resolution-independent: sqrt(w_rel * h_rel)
    px_at_640: float  # sqrt_area_px rescaled as if the image's long side became 640


def load_data_yaml(data_dir: Path) -> dict:
    with (data_dir / "data.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_samples(data_dir: Path, names: dict[int, str]) -> tuple[list[BoxSample], dict]:
    """Walk labels/{split}/*.txt, pairing each with its image to recover pixel sizes."""
    samples: list[BoxSample] = []
    images_per_split_class = {s: dict.fromkeys(names.values(), 0) for s in SPLITS}
    instances_per_split_class = {s: dict.fromkeys(names.values(), 0) for s in SPLITS}

    for split in SPLITS:
        label_dir = data_dir / "labels" / split
        image_dir = data_dir / "images" / split
        for label_path in sorted(label_dir.glob("*.txt")):
            with Image.open(image_dir / f"{label_path.stem}.jpg") as im:
                img_w, img_h = im.size

            classes_in_image: set[int] = set()
            for line in label_path.read_text(encoding="ascii").strip().splitlines():
                cls_str, cx, cy, w, h = line.split()
                cls_id = int(cls_str)
                w_rel, h_rel = float(w), float(h)
                width_px, height_px = w_rel * img_w, h_rel * img_h
                sqrt_area_px = math.sqrt(width_px * height_px)
                samples.append(
                    BoxSample(
                        split=split,
                        cls_id=cls_id,
                        width_px=width_px,
                        height_px=height_px,
                        sqrt_area_px=sqrt_area_px,
                        rel_sqrt_area=math.sqrt(w_rel * h_rel),
                        px_at_640=sqrt_area_px * TARGET_IMGSZ / max(img_w, img_h),
                    )
                )
                instances_per_split_class[split][names[cls_id]] += 1
                classes_in_image.add(cls_id)
            for cls_id in classes_in_image:
                images_per_split_class[split][names[cls_id]] += 1

    counts = {
        "images_per_split_class": images_per_split_class,
        "instances_per_split_class": instances_per_split_class,
    }
    return samples, counts


def _pct(values: list[float]) -> tuple[float, float, float]:
    p10, p50, p90 = np.percentile(np.asarray(values), PERCENTILES)
    return float(p10), float(p50), float(p90)


def plot_class_balance(counts: dict, names: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(names))
    width = 0.25
    for i, split in enumerate(SPLITS):
        values = [counts["images_per_split_class"][split][n] for n in names]
        ax.bar(x + (i - 1) * width, values, width, label=split)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("images")
    ax.set_title("Images per class per split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_histogram(
    values: list[float], out_path: Path, title: str, xlabel: str, ref_line: float | None = None
) -> None:
    p10, p50, p90 = _pct(values)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins=40, color="#3b7dd8", edgecolor="white")
    for p, val in (("p10", p10), ("p50 (median)", p50), ("p90", p90)):
        ax.axvline(val, color="#d84b3b", linestyle="--", linewidth=1)
        ax.text(val, ax.get_ylim()[1] * 0.95, f"{p}={val:.1f}", rotation=90, va="top", fontsize=8)
    if ref_line is not None:
        ax.axvline(ref_line, color="#666666", linestyle=":", linewidth=1)
        label = f"COCO small={ref_line:.0f}px (ref only)"
        ax.text(ref_line, ax.get_ylim()[1] * 0.5, label, rotation=90, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("instance count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_stats_md(
    out_dir: Path,
    data_dir: Path,
    names: list[str],
    counts: dict,
    samples: list[BoxSample],
) -> None:
    all_width = [s.width_px for s in samples]
    all_height = [s.height_px for s in samples]
    all_sqrt_area = [s.sqrt_area_px for s in samples]
    all_rel = [s.rel_sqrt_area * 100 for s in samples]  # as % of image diagonal-ish scale
    all_640 = [s.px_at_640 for s in samples]
    p10_640, p50_640, p90_640 = _pct(all_640)

    lines: list[str] = []
    lines.append("# EDA 統計報告\n")
    lines.append(f"資料來源：`{data_dir.as_posix()}`（{len(samples)} 個標註框）\n")

    lines.append("## 每類每 split 圖片／實例數\n")
    lines.append("| split | imgs | boxes | " + " | ".join(names) + " |")
    lines.append("|---|---|---|" + "---|" * len(names))
    for split in SPLITS:
        img_c = counts["images_per_split_class"][split]
        box_c = counts["instances_per_split_class"][split]
        n_imgs = len(list((data_dir / "labels" / split).glob("*.txt")))
        cells = " | ".join(f"{img_c[n]}/{box_c[n]}" for n in names)
        lines.append(f"| {split} | {n_imgs} | {sum(box_c.values())} | {cells} |")
    lines.append("\n(每格為 圖片數/框數)\n")

    lines.append("## Bbox 尺寸分布（整體，單位 px，除相對欄位外）\n")
    lines.append("| 指標 | p10 | p50（中位數） | p90 |")
    lines.append("|---|---|---|---|")
    for label, values, suffix in (
        ("width_px", all_width, ""),
        ("height_px", all_height, ""),
        ("sqrt_area_px（原始解析度）", all_sqrt_area, ""),
        ("相對面積開根號 (%)", all_rel, "%"),
    ):
        p10, p50, p90 = _pct(values)
        lines.append(f"| {label} | {p10:.1f}{suffix} | {p50:.1f}{suffix} | {p90:.1f}{suffix} |")

    lines.append("\n## Bbox 尺寸換算至 imgsz=640（決定性圖表）\n")
    lines.append("`box_px_at_640 = sqrt_area_px * 640 / max(原始寬, 原始高)`\n")
    lines.append("| p10 | p50（中位數） | p90 |")
    lines.append("|---|---|---|")
    lines.append(f"| {p10_640:.1f}px | {p50_640:.1f}px | {p90_640:.1f}px |\n")

    lines.append("### 各類別中位數（換算至 640）\n")
    lines.append("| " + " | ".join(names) + " |")
    lines.append("|" + "---|" * len(names))
    per_class_medians = []
    for i in range(len(names)):
        vals = [s.px_at_640 for s in samples if s.cls_id == i]
        per_class_medians.append(_pct(vals)[1] if vals else float("nan"))
    lines.append("| " + " | ".join(f"{v:.1f}px" for v in per_class_medians) + " |\n")

    lines.append("## imgsz / SAHI 論證\n")
    lines.append(
        f"實測結果：全資料集瑕疵框換算至 imgsz=640 後，**中位數約 {p50_640:.1f}px**"
        f"（p10={p10_640:.1f}px, p90={p90_640:.1f}px），全部落在小物件偵測的困難區間"
        f"（COCO 的小物件門檻 32px 僅供對照，非本資料集標準——這裡連 p90 都遠低於它）。"
        "這支持兩個決策：(1) `imgsz=640` 可作為訓練基礎解析度，但屬於「可行但吃緊」的選擇，"
        "YOLO26 的 STAL（Small-Target-Aware Label assignment）正是針對此類小物件場景設計，"
        "不對抗這個困難、而是選擇本來就為此設計的架構；(2) Phase 2 的 SAHI 切片推論實驗"
        "動機正是量化「把瑕疵在推論時放大回接近原始解析度」能挽回多少 recall——"
        f"本節測得的中位數 {p50_640:.1f}px 就是 SAHI 實驗要對照、企圖改善的基準數字。"
    )

    (out_dir / "stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EDA: bbox size distributions and class balance.")
    parser.add_argument("--data", default="data/pcb")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args(argv)

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = load_data_yaml(data_dir)
    names = [data_yaml["names"][i] for i in sorted(data_yaml["names"])]

    samples, counts = collect_samples(data_dir, data_yaml["names"])

    plot_class_balance(counts, names, out_dir / "class_balance.png")
    plot_histogram(
        [s.rel_sqrt_area * 100 for s in samples],
        out_dir / "bbox_size_relative.png",
        "Relative bbox size (resolution-independent)",
        "sqrt(area) as % of image area",
    )
    plot_histogram(
        [s.px_at_640 for s in samples],
        out_dir / "bbox_size_at_640.png",
        "Bbox size rescaled to imgsz=640 (long side)",
        "sqrt(area) in px at imgsz=640",
        ref_line=COCO_SMALL_PX,
    )
    write_stats_md(out_dir, data_dir, names, counts, samples)

    n_images = sum(len(list((data_dir / "labels" / s).glob("*.txt"))) for s in SPLITS)
    print(f"{len(samples)} boxes analyzed across {n_images} images")
    print(f"stats written to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

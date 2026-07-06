"""VOC records -> YOLO dataset layout (images/, labels/, data.yaml)."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from pcb_defect.constants import CLASSES
from pcb_defect.data_prep.split import SPLITS, SplitResult
from pcb_defect.data_prep.voc import VocBox, VocRecord


class ConversionError(Exception):
    """A YOLO label value fell outside its valid range."""


def yolo_line(box: VocBox, width: int, height: int) -> str:
    """One YOLO label line: `cls cx cy w h`, normalized, 6 decimals."""
    cx = (box.xmin + box.xmax) / 2.0 / width
    cy = (box.ymin + box.ymax) / 2.0 / height
    bw = (box.xmax - box.xmin) / width
    bh = (box.ymax - box.ymin) / height
    for name, value in (("cx", cx), ("cy", cy), ("w", bw), ("h", bh)):
        if not 0.0 < value <= 1.0:
            raise ConversionError(f"{name}={value} out of (0, 1] for box {box}")
    return f"{box.cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def convert_dataset(records: list[VocRecord], split: SplitResult, out_dir: Path) -> dict:
    """Copy images, write YOLO labels and data.yaml. Wipes out_dir first (idempotent).

    Images are copied, not symlinked - Windows symlinks require Developer Mode.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for s in SPLITS:
        (out_dir / "images" / s).mkdir(parents=True)
        (out_dir / "labels" / s).mkdir(parents=True)

    images = dict.fromkeys(SPLITS, 0)
    instances = {s: dict.fromkeys(CLASSES, 0) for s in SPLITS}
    image_classes = {s: dict.fromkeys(CLASSES, 0) for s in SPLITS}
    warnings: list[dict] = []

    for r in sorted(records, key=lambda rec: rec.stem):
        s = split.assignment[r.stem]
        shutil.copy2(r.image_path, out_dir / "images" / s / r.image_path.name)
        lines = [yolo_line(b, r.width, r.height) for b in r.boxes]
        label_path = out_dir / "labels" / s / f"{r.stem}.txt"
        label_path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
        images[s] += 1
        for b in r.boxes:
            instances[s][CLASSES[b.cls_id]] += 1
        for cls_id in {b.cls_id for b in r.boxes}:
            image_classes[s][CLASSES[cls_id]] += 1
        warnings.extend({"file": r.image_path.name, "warning": w} for w in r.warnings)

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASSES)),
    }
    # allow_unicode: the absolute path may contain CJK characters on this machine
    # newline="\n": pin LF so the same seed produces byte-identical output on Windows and Colab
    with (out_dir / "data.yaml").open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    return {
        "images_total": sum(images.values()),
        "boxes_total": sum(sum(v.values()) for v in instances.values()),
        "images_per_split": images,
        "image_count_per_split_class": image_classes,
        "instances_per_split_class": instances,
        "warnings": warnings,
    }

"""CLI orchestrator: download -> parse -> tripwires -> split -> convert -> reports.

Used identically on the local machine and inside the Colab notebook:

    uv run python -m pcb_defect.data_prep.prepare --out data/pcb --strategy grouped --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pcb_defect.constants import (
    CLASSES,
    EXPECTED_BOARDS,
    EXPECTED_BOXES,
    EXPECTED_IMAGES,
    SEED,
)
from pcb_defect.data_prep.convert import convert_dataset
from pcb_defect.data_prep.download import download_raw, validate_layout
from pcb_defect.data_prep.split import SPLITS, SplitResult, board_id, grouped_split, random_split
from pcb_defect.data_prep.voc import VocRecord, parse_voc_xml


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.raw_dir:
        raw_root = validate_layout(Path(args.raw_dir))
    else:
        raw_root = download_raw(force=args.force_download)
    print(f"raw dataset: {raw_root}")

    xml_paths = sorted((raw_root / "Annotations").rglob("*.xml"))
    images_root = raw_root / "images"
    records = [parse_voc_xml(p, images_root) for p in xml_paths]
    _check_tripwires(records)

    if args.strategy == "grouped":
        split = grouped_split(records, seed=args.seed)
    else:
        split = random_split(records, seed=args.seed)

    out_dir = Path(args.out)
    report = convert_dataset(records, split, out_dir)
    _write_reports(report, split, out_dir)
    _print_summary(report, split)
    print(f"\ndataset written to {out_dir.resolve()}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pcb_defect.data_prep.prepare",
        description="Download HRIPCB, convert VOC->YOLO, split and write reports.",
    )
    parser.add_argument("--out", default="data/pcb", help="output dir (wiped and rebuilt)")
    parser.add_argument("--strategy", choices=("grouped", "random"), default="grouped")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--raw-dir", default=None, help="existing download (skips kagglehub)")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args(argv)


def _check_tripwires(records: list[VocRecord]) -> None:
    """Abort if the download stops matching the empirically verified dataset."""
    n_boxes = sum(len(r.boxes) for r in records)
    boards = {board_id(r.stem) for r in records}
    problems = []
    if len(records) != EXPECTED_IMAGES:
        problems.append(f"images {len(records)} != {EXPECTED_IMAGES}")
    if n_boxes != EXPECTED_BOXES:
        problems.append(f"boxes {n_boxes} != {EXPECTED_BOXES}")
    if len(boards) != EXPECTED_BOARDS:
        problems.append(f"boards {len(boards)} != {EXPECTED_BOARDS}")
    if problems:
        sys.exit("dataset tripwire failed: " + "; ".join(problems))


def _write_reports(report: dict, split: SplitResult, out_dir: Path) -> None:
    split_report = {
        "strategy": split.strategy,
        "seed_requested": split.seed_requested,
        "seed_used": split.seed_used,
        "board_to_split": split.board_to_split,
        "images_per_split": report["images_per_split"],
        "image_count_per_split_class": report["image_count_per_split_class"],
        "instances_per_split_class": report["instances_per_split_class"],
    }
    conversion_report = {
        "images_total": report["images_total"],
        "boxes_total": report["boxes_total"],
        "warning_count": len(report["warnings"]),
        "warnings": report["warnings"],
    }
    # newline="\n": pin LF so the same seed produces byte-identical reports on Windows and Colab
    (out_dir / "split_report.json").write_text(
        json.dumps(split_report, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    (out_dir / "conversion_report.json").write_text(
        json.dumps(conversion_report, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )


def _print_summary(report: dict, split: SplitResult) -> None:
    print(f"\nstrategy={split.strategy}  seed_used={split.seed_used}", end="")
    if split.seed_used != split.seed_requested:
        print(f"  (requested {split.seed_requested}, retried for class coverage)", end="")
    print()
    if split.board_to_split:
        by_split: dict[str, list[str]] = {}
        for board, s in sorted(split.board_to_split.items()):
            by_split.setdefault(s, []).append(board)
        for s in SPLITS:
            print(f"  {s:<5} boards: {', '.join(by_split.get(s, []))}")

    print(f"\n{'split':<6}{'imgs':>6}{'boxes':>7}" + "".join(f"{c[:14]:>16}" for c in CLASSES))
    for s in SPLITS:
        img_c = report["image_count_per_split_class"][s]
        box_c = report["instances_per_split_class"][s]
        cells = "".join(f"{f'{img_c[c]}/{box_c[c]}':>16}" for c in CLASSES)
        print(f"{s:<6}{report['images_per_split'][s]:>6}{sum(box_c.values()):>7}" + cells)
    totals = f"{report['images_total']} images / {report['boxes_total']} boxes"
    print(f"total: {totals}  (cells: imgs/boxes)")

    warnings = report["warnings"]
    print(f"warnings: {len(warnings)}")
    for w in warnings[:5]:
        print(f"  - {w['file']}: {w['warning']}")
    if len(warnings) > 5:
        print(f"  ... and {len(warnings) - 5} more (see conversion_report.json)")


if __name__ == "__main__":
    raise SystemExit(main())

"""Board-grouped split invariants: partition, determinism, non-contiguous board ids."""

from collections import Counter
from pathlib import Path

import pytest

from pcb_defect.data_prep.split import SplitError, board_id, grouped_split, random_split
from pcb_defect.data_prep.voc import VocBox, VocRecord

# Non-contiguous on purpose - the real dataset starts 01, 04, 05, ...
BOARDS = ["01", "04", "05", "06", "07", "08", "09", "10", "11", "12"]


def _records() -> list[VocRecord]:
    records = []
    for board in BOARDS:
        for cls in range(6):
            for i in range(2):
                stem = f"{board}_class{cls}_{i:02d}"
                records.append(
                    VocRecord(
                        image_path=Path(f"{stem}.jpg"),
                        width=100,
                        height=100,
                        boxes=[VocBox(cls, 1, 1, 10, 10)],
                    )
                )
    return records


def test_grouped_split_is_board_partition_and_deterministic():
    records = _records()
    first = grouped_split(records, seed=42)
    second = grouped_split(records, seed=42)

    assert first.assignment == second.assignment  # same seed -> identical split
    assert first.board_to_split == second.board_to_split
    assert set(first.assignment) == {r.stem for r in records}  # covers every image

    # a board never crosses splits
    for r in records:
        assert first.assignment[r.stem] == first.board_to_split[board_id(r.stem)]

    assert Counter(first.board_to_split.values()) == {"train": 8, "val": 1, "test": 1}


def test_board_id_noncontiguous():
    assert board_id("04_spur_07") == "04"
    with pytest.raises(SplitError):
        board_id("spur_07")


def test_random_split_covers_every_class_in_every_split():
    records = _records()  # 12 boards x 6 classes x 2 images = 24 images per class
    result = random_split(records, seed=42)

    assert set(result.assignment) == {r.stem for r in records}
    per_split_classes = {"train": set(), "val": set(), "test": set()}
    for r in records:
        per_split_classes[result.assignment[r.stem]].add(r.boxes[0].cls_id)
    all_classes = set(range(6))
    assert per_split_classes["train"] == all_classes
    assert per_split_classes["val"] == all_classes
    assert per_split_classes["test"] == all_classes


def test_random_split_does_not_starve_train_for_small_class():
    # a 3-image class must still yield exactly 1/1/1, never 0 images in train
    records = [
        VocRecord(Path(f"99_class0_{i:02d}.jpg"), 100, 100, [VocBox(0, 1, 1, 10, 10)])
        for i in range(3)
    ]
    result = random_split(records, seed=42)
    assert Counter(result.assignment.values()) == {"train": 1, "val": 1, "test": 1}


def test_random_split_rejects_class_too_small_to_cover_all_splits():
    records = [
        VocRecord(Path(f"99_class0_{i:02d}.jpg"), 100, 100, [VocBox(0, 1, 1, 10, 10)])
        for i in range(2)
    ]
    with pytest.raises(SplitError):
        random_split(records, seed=42)

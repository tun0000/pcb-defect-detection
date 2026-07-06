"""Train/val/test split strategies with board-level leakage control."""

from __future__ import annotations

import random
from dataclasses import dataclass

from pcb_defect.constants import BOARD_ID_RE, SEED
from pcb_defect.data_prep.voc import VocRecord

SPLITS = ("train", "val", "test")


class SplitError(Exception):
    """No valid split could be produced."""


@dataclass
class SplitResult:
    strategy: str
    seed_requested: int
    seed_used: int
    assignment: dict[str, str]  # image stem -> split
    board_to_split: dict[str, str] | None  # grouped strategy only


def board_id(stem: str) -> str:
    """Template-board id = leading number of the filename (non-contiguous: 01, 04, 05...)."""
    m = BOARD_ID_RE.match(stem)
    if m is None:
        raise SplitError(f"filename {stem!r} has no leading board id")
    return m.group(1)


def grouped_split(
    records: list[VocRecord], seed: int = SEED, max_attempts: int = 20
) -> SplitResult:
    """Primary anti-leakage split: whole template boards go to train/val/test (8/1/1).

    HRIPCB synthesizes all defects onto 10 physical boards; an image-level random
    split would put near-identical board backgrounds on both sides of the fence.
    Retries seed+1, seed+2, ... until every class appears in every split.
    """
    boards = sorted({board_id(r.stem) for r in records})
    if len(boards) < 3:
        raise SplitError(f"need at least 3 boards to split, found {len(boards)}")
    for offset in range(max_attempts):
        seed_used = seed + offset
        order = list(boards)
        random.Random(seed_used).shuffle(order)
        board_to_split = dict.fromkeys(order[:-2], "train")
        board_to_split[order[-2]] = "val"
        board_to_split[order[-1]] = "test"
        assignment = {r.stem: board_to_split[board_id(r.stem)] for r in records}
        if _classes_complete(records, assignment):
            return SplitResult("grouped", seed, seed_used, assignment, board_to_split)
    raise SplitError(f"no grouped split satisfied class coverage in {max_attempts} attempts")


def random_split(
    records: list[VocRecord],
    seed: int = SEED,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> SplitResult:
    """Literature-comparable image-level split, stratified by the image's (single) class.

    Leaks board backgrounds across splits by construction - it exists only to
    quantify that inflation against the grouped split.
    """
    rng = random.Random(seed)
    by_class: dict[int, list[VocRecord]] = {}
    for r in records:
        by_class.setdefault(r.boxes[0].cls_id, []).append(r)
    assignment: dict[str, str] = {}
    for cls in sorted(by_class):
        group = sorted(by_class[cls], key=lambda r: r.stem)
        if len(group) < 3:
            raise SplitError(f"class {cls} has only {len(group)} images, need >=3 to cover splits")
        rng.shuffle(group)
        # clamp so a small class can never starve train/test of their >=1 guarantee
        n_val = max(1, min(round(len(group) * ratios[1]), len(group) - 2))
        n_test = max(1, min(round(len(group) * ratios[2]), len(group) - n_val - 1))
        mid = n_val + n_test
        for r in group[:n_val]:
            assignment[r.stem] = "val"
        for r in group[n_val:mid]:
            assignment[r.stem] = "test"
        for r in group[mid:]:
            assignment[r.stem] = "train"
    if not _classes_complete(records, assignment):
        raise SplitError("random split failed class coverage check")
    return SplitResult("random", seed, seed, assignment, None)


def _classes_complete(records: list[VocRecord], assignment: dict[str, str]) -> bool:
    """Every class that exists in the dataset must appear in every split."""
    universe = {b.cls_id for r in records for b in r.boxes}
    seen: dict[str, set[int]] = {s: set() for s in SPLITS}
    for r in records:
        seen[assignment[r.stem]].update(b.cls_id for b in r.boxes)
    return all(seen[s] == universe for s in SPLITS)

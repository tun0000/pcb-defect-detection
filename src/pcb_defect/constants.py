"""Project-wide constants: the single source of truth for classes, seed and dataset identity.

data.yaml, the tests and the Phase-2 ONNX demo must all derive from these values -
never redefine them elsewhere.
"""

from __future__ import annotations

import re

KAGGLE_DATASET = "akhatova/pcb-defects"

# Canonical class order (alphabetical), ids 0-5.
# XML <name> tags are lowercase_with_underscores; folder names are Capitalized_first -
# always normalize case-insensitively before lookup.
CLASSES: list[str] = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]
CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

SEED = 42

# HRIPCB filename prefix = template-board id, e.g. "01_missing_hole_05" -> "01".
# Board ids are non-contiguous (01, 04, 05, ...) - never assume 1..10.
BOARD_ID_RE = re.compile(r"^(\d+)_")

# Dataset tripwires (empirically verified against the Kaggle version, 2026-07-06).
# If the download stops matching these, abort instead of silently converting something else.
EXPECTED_IMAGES = 693
EXPECTED_BOXES = 2953
EXPECTED_BOARDS = 10

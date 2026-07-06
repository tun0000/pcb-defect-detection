"""Anonymous Kaggle download with layout tripwires."""

from __future__ import annotations

from pathlib import Path

from pcb_defect.constants import EXPECTED_IMAGES, KAGGLE_DATASET


class DatasetLayoutError(Exception):
    """The downloaded dataset does not match the verified HRIPCB layout."""


def download_raw(force: bool = False) -> Path:
    """Download akhatova/pcb-defects via kagglehub (anonymous) and return .../PCB_DATASET.

    One-shot full download only: anonymous per-file requests get rate-limited fast.
    The ~2 GB zip is cached under ~/.cache/kagglehub and reused on re-runs.
    """
    import kagglehub  # deferred so pytest never touches network-related imports

    path = Path(kagglehub.dataset_download(KAGGLE_DATASET, force_download=force))
    return validate_layout(path)


def validate_layout(download_root: Path) -> Path:
    """Locate PCB_DATASET/ and enforce the 693/693 tripwire before any conversion."""
    root = download_root / "PCB_DATASET"
    if not root.is_dir():
        if (download_root / "Annotations").is_dir():  # --raw-dir may point at PCB_DATASET itself
            root = download_root
        else:
            raise DatasetLayoutError(f"PCB_DATASET/ not found under {download_root}")
    n_xml = sum(1 for _ in (root / "Annotations").rglob("*.xml"))
    n_img = sum(1 for _ in (root / "images").rglob("*.jpg"))
    if n_xml != EXPECTED_IMAGES or n_img != EXPECTED_IMAGES:
        raise DatasetLayoutError(
            f"tripwire: expected {EXPECTED_IMAGES} xmls and jpgs under Annotations/ and images/, "
            f"found {n_xml} xmls / {n_img} jpgs - the dataset may have changed, aborting"
        )
    return root

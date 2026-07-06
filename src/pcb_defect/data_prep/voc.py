"""Pascal VOC XML parsing and validation for the HRIPCB dataset."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from pcb_defect.constants import CLASS_TO_ID

# A clamp that moves an edge by more than this many pixels is suspicious enough to log.
CLAMP_WARN_PX = 2.0
# Boxes thinner than this (px) after clamping carry no signal and are dropped.
MIN_BOX_PX = 1.0


class VocError(Exception):
    """Hard error while parsing VOC annotations - aborts the conversion."""


class UnknownClassError(VocError):
    """XML <name> not in the canonical class list."""


@dataclass
class VocBox:
    cls_id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class VocRecord:
    image_path: Path
    width: int
    height: int
    boxes: list[VocBox]
    warnings: list[str] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return self.image_path.stem


def normalize_class(raw: str) -> int:
    """Map an XML <name> to a class id, case-insensitively.

    HRIPCB folder names are Capitalized_first while XML <name> tags are lowercase;
    both must resolve to the same id.
    """
    key = raw.strip().lower()
    try:
        return CLASS_TO_ID[key]
    except KeyError:
        raise UnknownClassError(f"unknown class name: {raw!r}") from None


def parse_voc_xml(xml_path: Path, images_root: Path) -> VocRecord:
    """Parse and validate a single HRIPCB VOC annotation.

    The image path is derived from the file layout
    (Annotations/<Class>/<stem>.xml -> images/<Class>/<stem>.jpg).
    <filename> is only cross-checked; <folder>/<path> are the uploader's local
    paths and are ignored entirely.
    """
    warnings: list[str] = []
    root = ET.parse(xml_path).getroot()

    image_path = images_root / xml_path.parent.name / f"{xml_path.stem}.jpg"
    if not image_path.is_file():
        raise VocError(f"{xml_path.name}: derived image not found: {image_path}")

    declared = (root.findtext("filename") or "").strip()
    if declared and Path(declared).stem != xml_path.stem:
        warnings.append(f"<filename> {declared!r} does not match xml stem {xml_path.stem!r}")

    width = _to_int(root.findtext("size/width"))
    height = _to_int(root.findtext("size/height"))
    if width <= 0 or height <= 0:
        with Image.open(image_path) as im:
            width, height = im.size
        warnings.append(f"<size> missing or invalid, read {width}x{height} from the image")

    boxes: list[VocBox] = []
    for obj in root.iter("object"):
        cls_id = normalize_class(obj.findtext("name") or "")
        bb = obj.find("bndbox")
        if bb is None:
            raise VocError(f"{xml_path.name}: <object> without <bndbox>")
        try:
            coords = [float(bb.findtext(tag)) for tag in ("xmin", "ymin", "xmax", "ymax")]
        except (TypeError, ValueError) as exc:
            raise VocError(f"{xml_path.name}: malformed <bndbox>") from exc
        xmin, ymin, xmax, ymax = coords

        cxmin, cxmax = _clamp(xmin, width), _clamp(xmax, width)
        cymin, cymax = _clamp(ymin, height), _clamp(ymax, height)
        shift = max(abs(cxmin - xmin), abs(cxmax - xmax), abs(cymin - ymin), abs(cymax - ymax))
        if shift > CLAMP_WARN_PX:
            warnings.append(f"box clamped by {shift:.1f}px: ({xmin}, {ymin}, {xmax}, {ymax})")
        if cxmax - cxmin < MIN_BOX_PX or cymax - cymin < MIN_BOX_PX:
            warnings.append(f"degenerate box dropped: ({xmin}, {ymin}, {xmax}, {ymax})")
            continue
        boxes.append(VocBox(cls_id, cxmin, cymin, cxmax, cymax))

    if not boxes:
        raise VocError(f"{xml_path.name}: no usable boxes (every HRIPCB image has 3-6)")

    return VocRecord(
        image_path=image_path, width=width, height=height, boxes=boxes, warnings=warnings
    )


def _to_int(text: str | None) -> int:
    try:
        return int(float(text))  # tolerate VOC dumps that write "3034.0"
    except (TypeError, ValueError):
        return 0


def _clamp(value: float, upper: int) -> float:
    return min(max(value, 0.0), float(upper))

"""VOC parsing/validation and YOLO conversion tests against the handcrafted fixture."""

from pathlib import Path

import pytest

from pcb_defect.data_prep.convert import yolo_line
from pcb_defect.data_prep.voc import UnknownClassError, VocBox, normalize_class, parse_voc_xml

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def hripcb_layout(tmp_path):
    """Rebuild the real HRIPCB folder shape around the committed fixture files."""
    xml_dir = tmp_path / "Annotations" / "Missing_hole"
    img_dir = tmp_path / "images" / "Missing_hole"
    xml_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)
    xml = xml_dir / "01_missing_hole_01.xml"
    xml.write_bytes((FIXTURES / "sample.xml").read_bytes())
    (img_dir / "01_missing_hole_01.jpg").write_bytes((FIXTURES / "sample.jpg").read_bytes())
    return xml, tmp_path / "images"


def test_parse_validate_and_normalize(hripcb_layout):
    xml, images_root = hripcb_layout
    rec = parse_voc_xml(xml, images_root)

    assert (rec.width, rec.height) == (64, 48)
    # 4 objects in, 3 boxes out: the degenerate zero-size box is dropped
    assert [b.cls_id for b in rec.boxes] == [0, 1, 0]  # Mouse_bite casing normalized to id 1
    assert rec.boxes[2].xmax == 64.0  # xmax=70 clamped to image width

    assert len(rec.warnings) == 3
    joined = "\n".join(rec.warnings)
    assert "does not match" in joined  # <filename> deliberately mismatches the stem
    assert "clamped" in joined
    assert "degenerate" in joined


def test_yolo_line_golden():
    # (10,10,30,30) in a 64x48 image, hand-computed to 6 decimals
    assert yolo_line(VocBox(0, 10, 10, 30, 30), 64, 48) == "0 0.312500 0.416667 0.312500 0.416667"


def test_unknown_class_raises():
    assert normalize_class("Missing_Hole") == 0  # case-insensitive normalization
    with pytest.raises(UnknownClassError):
        normalize_class("scratch")

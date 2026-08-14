"""Unit tests for Block 3 mark classification, digit/date helpers, and prior fusion."""

from __future__ import annotations

import numpy as np

from med_doc.htr.fusion import fuse_handwriting
from med_doc.htr.marks import classify_mark, ink_density
from med_doc.htr.recognizer import extract_digits, has_ink, parse_datetime
from med_doc.kg import KnowledgeGraph
from med_doc.paths import DEFAULT_KG


def _empty_checkbox(size: int = 32) -> np.ndarray:
    img = np.full((size, size, 3), 245, dtype=np.uint8)
    img[0:2, :] = 20
    img[-2:, :] = 20
    img[:, 0:2] = 20
    img[:, -2:] = 20
    return img


def _ticked_checkbox(size: int = 32) -> np.ndarray:
    img = _empty_checkbox(size)
    for i in range(6, size - 6):
        img[i, i] = 15
        if i + 1 < size - 6:
            img[i, i + 1] = 15
    return img


def _filled_checkbox(size: int = 32) -> np.ndarray:
    img = _empty_checkbox(size)
    img[6:-6, 6:-6] = 25
    return img


def test_empty_checkbox_is_unmarked():
    pred = classify_mark(_empty_checkbox(), "cbc")
    assert pred.is_marked is False
    assert pred.needs_hitl is False
    assert pred.ink_density < 0.08


def test_ticked_checkbox_is_marked():
    pred = classify_mark(_ticked_checkbox(), "cbc")
    assert pred.is_marked is True
    assert pred.ink_density >= 0.05
    assert pred.source in {"slash", "density", "filled"}


def test_filled_checkbox_is_marked():
    pred = classify_mark(_filled_checkbox(), "alt")
    assert pred.is_marked is True
    assert pred.source == "filled"
    assert pred.confidence >= 0.9


def test_metadata_fallback_without_crop():
    pred = classify_mark(None, "cbc", fallback_dark_ratio=0.28, fallback_candidate=True)
    assert pred.is_marked is True
    assert pred.source == "metadata_fallback"


def test_extract_digits_and_date_parse():
    assert extract_digits("x2") == "2"
    assert extract_digits("EDTA 1") == "1"
    parsed, conf = parse_datetime("14/08/2026 09:30")
    assert parsed.startswith("14/08/2026")
    assert "09:30" in parsed
    assert conf >= 0.8


def test_has_ink_detects_stroke():
    assert has_ink(_ticked_checkbox()) is True
    assert has_ink(_empty_checkbox(), min_frac=0.15) is False


def test_fuse_others_with_kg_prior():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    fused = fuse_handwriting(
        "others",
        "triglyc",
        0.55,
        "trocr",
        kg=kg,
        ticked_ids=["profile_lipid"],
    )
    assert fused.canonical_id == "triglycerides"
    assert fused.canonical_value is not None


def test_fuse_tube_matches_expected():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    fused = fuse_handwriting(
        "tube_edta",
        "1",
        0.8,
        "trocr",
        kg=kg,
        ticked_ids=["cbc"],
    )
    assert fused.canonical_value == "1"
    assert fused.needs_hitl is False
    assert fused.confidence >= 0.9


def test_fuse_empty_others_no_hitl():
    fused = fuse_handwriting("others", "", 0.85, "empty")
    assert fused.canonical_value is None
    assert fused.needs_hitl is False
    assert fused.source == "empty"


def test_ink_density_range():
    empty = ink_density(_empty_checkbox())
    filled = ink_density(_filled_checkbox())
    assert 0.0 <= empty < 0.15
    assert filled > empty

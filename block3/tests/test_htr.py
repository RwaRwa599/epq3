"""Standalone Block 3 unit tests (no Block 1 template dependency)."""

from __future__ import annotations

import numpy as np

from med_doc.htr.fusion import fuse_handwriting
from med_doc.htr.marks import classify_mark
from med_doc.htr.recognizer import extract_digits, parse_datetime
from med_doc.kg import KnowledgeGraph
from med_doc.paths import DEFAULT_KG


def _empty_checkbox(size: int = 32) -> np.ndarray:
    img = np.full((size, size, 3), 245, dtype=np.uint8)
    img[0:2, :] = 20
    img[-2:, :] = 20
    img[:, 0:2] = 20
    img[:, -2:] = 20
    return img


def _filled_checkbox(size: int = 32) -> np.ndarray:
    img = _empty_checkbox(size)
    img[6:-6, 6:-6] = 25
    return img


def test_empty_and_filled_marks():
    empty = classify_mark(_empty_checkbox(), "cbc")
    filled = classify_mark(_filled_checkbox(), "alt")
    assert empty.is_marked is False
    assert filled.is_marked is True


def test_digits_and_dates():
    assert extract_digits("x2") == "2"
    parsed, conf = parse_datetime("14/08/2026 09:30")
    assert "14/08/2026" in parsed
    assert conf >= 0.8


def test_fuse_with_kg():
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

    tube = fuse_handwriting("tube_edta", "1", 0.8, "trocr", kg=kg, ticked_ids=["cbc"])
    assert tube.canonical_value == "1"
    assert tube.needs_hitl is False

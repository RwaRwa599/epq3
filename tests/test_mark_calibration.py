"""Crop-level gold, logreg refit, crop QA, model inventory (no PHI)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from med_doc.eval.calibrate_marks import fit_mark_logreg, train_and_save, _empty, _slash
from med_doc.eval.crop_qa import diagnose_crop
from med_doc.eval.export_crops import export_labeled_crops
from med_doc.eval.mark_gold import GoldSet, SheetGold, load_gold
from med_doc.eval.models import models_in_use
from med_doc.htr.crop_cnn import cnn_available
from med_doc.htr import marks as marks_mod
from med_doc.htr.marks import classify_mark
from med_doc.htr.nonverbal import classify_mark_nonverbal
from test_block3_verbal_nonverbal import _write_mini_block1


def test_example_gold_ca125_only():
    gold = load_gold(Path("data/labels/examples/IMG_7596.json"))
    assert gold.sheets[0].ticked_field_ids == ["ca125"]


def test_export_crops_splits_empty_and_tick(tmp_path: Path):
    b1 = _write_mini_block1(tmp_path / "b1")
    gold = GoldSet(
        sheets=[SheetGold(doc_id="sample_sheet_01", ticked_field_ids=["cbc"])]
    )
    out = tmp_path / "crops"
    stats = export_labeled_crops(b1, gold, out)
    assert stats["tick"] == 1
    assert stats["empty"] == 1
    assert (out / "sample_sheet_01" / "tick" / "cbc.png").is_file()
    assert (out / "sample_sheet_01" / "empty" / "alt.png").is_file()


def test_refit_logreg_keeps_empty_unmarked(tmp_path: Path):
    dest = tmp_path / "mark_weights.json"
    prev_w = marks_mod.LOGREG_W.copy()
    prev_b = float(marks_mod.LOGREG_B)
    try:
        stats = train_and_save(dest, crop_dir=None, empty_repeats=6)
        assert dest.is_file()
        assert stats["empty_marked"] is False
        assert stats["slash_marked"] is True
        payload = json.loads(dest.read_text())
        assert len(payload["w"]) == 8
        assert classify_mark(_empty(), "alt").is_marked is False
        assert classify_mark(_slash(), "cbc").is_marked is True
    finally:
        marks_mod.LOGREG_W = prev_w
        marks_mod.LOGREG_B = prev_b


def test_fit_mixes_empties_more_than_ticks():
    w, b, stats = fit_mark_logreg(empty_repeats=8)
    assert stats["n_empty"] >= stats["n_tick"]
    assert stats["n_rows"] >= 12
    assert w.shape == (8,)
    assert isinstance(b, float)


def test_gold_tick_on_label_strip_is_block1_shift():
    img = np.full((20, 80, 3), 250, dtype=np.uint8)
    img[8:14, 4:-4] = 40
    row = diagnose_crop(img, gold_ticked=True, field_id="ca125")
    assert row["cause"] == "block1_shift"
    assert row["hollow_ring"] is False


def test_models_report_geometry_not_paddle_or_cnn():
    report = models_in_use(verbal_backend="lexicon", mark_backend="geometry")
    assert report["paddle_used_for_ticks"] is False
    assert report["trocr_used"] is False
    assert "logreg" in report["ticks"]
    assert cnn_available() is False


def test_nonverbal_default_skips_paddle(monkeypatch):
    called = {"n": 0}

    def boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("Paddle must not run on the default tick path")

    monkeypatch.setattr("med_doc.htr.nonverbal._get_paddle", boom)
    pred = classify_mark_nonverbal(_slash(), "cbc")
    assert pred.is_marked is True
    assert called["n"] == 0

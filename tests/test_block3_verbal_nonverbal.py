"""Block 3: nonverbal marks, verbal HTR, Block 1 ZIP ingest, Block 2 KG import."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from med_doc.htr.ingest import load_block1_document, unzip_or_dir
from med_doc.htr.nonverbal import classify_mark_nonverbal, classify_marks, paddle_available
from med_doc.htr.verbal import recognize_fields, recognize_verbal, trocr_available
from med_doc.htr.batch import process_from_block1
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


def _write_mini_block1(root: Path) -> Path:
    """Tiny Block 1 ZIP: one ticked CBC, empty ALT, empty others/tube_edta."""
    doc = root / "docs" / "sample_sheet_01"
    (doc / "crops" / "checkboxes").mkdir(parents=True)
    (doc / "crops" / "handwriting").mkdir(parents=True)
    import cv2

    canvas = np.full((200, 300, 3), 245, dtype=np.uint8)
    cv2.imwrite(str(doc / "canonical.png"), canvas)
    empty = _empty_checkbox()
    tick = _ticked_checkbox()
    cv2.imwrite(str(doc / "crops" / "checkboxes" / "cbc.png"), tick)
    cv2.imwrite(str(doc / "crops" / "checkboxes" / "alt.png"), empty)
    cv2.imwrite(str(doc / "crops" / "handwriting" / "others.png"), empty)
    cv2.imwrite(str(doc / "crops" / "handwriting" / "tube_edta.png"), empty)
    meta = {
        "doc_id": "sample_sheet_01",
        "fields": {
            "checkboxes": {
                "cbc": {"bbox": [10, 10, 42, 42], "crop_path": "crops/checkboxes/cbc.png"},
                "alt": {"bbox": [50, 10, 82, 42], "crop_path": "crops/checkboxes/alt.png"},
            },
            "handwriting": {
                "others": {"bbox": [10, 60, 120, 90], "crop_path": "crops/handwriting/others.png"},
                "tube_edta": {"bbox": [10, 100, 80, 130], "crop_path": "crops/handwriting/tube_edta.png"},
            },
        },
        "detected_marks": {
            "cbc": {"dark_ratio": 0.22, "is_marked_candidate": True},
            "alt": {"dark_ratio": 0.02, "is_marked_candidate": False},
        },
    }
    (doc / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "block": "block1",
                "total_documents": 1,
                "documents": [{"doc_id": "sample_sheet_01", "status": "success"}],
            }
        ),
        encoding="utf-8",
    )
    zip_path = root / "block1_normalized_batch.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for fp in root.rglob("*"):
            if fp.is_file() and fp != zip_path:
                zf.write(fp, arcname=fp.relative_to(root))
    return zip_path


def test_nonverbal_empty_and_ticked_without_paddle():
    empty = classify_mark_nonverbal(_empty_checkbox(), "alt")
    tick = classify_mark_nonverbal(_ticked_checkbox(), "cbc")
    filled = classify_mark_nonverbal(_filled_checkbox(), "profile_lipid")
    assert empty.is_marked is False
    assert tick.is_marked is True
    assert filled.is_marked is True
    if not paddle_available():
        assert empty.source == "density_fallback"
        assert tick.source == "density_fallback"
        assert filled.source == "density_fallback"
    else:
        assert empty.source in {"paddle", "density_fallback"}
        assert tick.source in {"paddle", "density_fallback"}


def test_nonverbal_batch_independent_of_trocr():
    marks = classify_marks({"cbc": _ticked_checkbox(), "alt": _empty_checkbox()})
    assert marks["cbc"].is_marked is True
    assert marks["alt"].is_marked is False
    assert set(marks) == {"cbc", "alt"}


def test_verbal_empty_crop_without_trocr():
    pred = recognize_verbal(_empty_checkbox(), "others", backend="trocr")
    assert pred.raw_text == ""
    assert pred.source == "empty"
    assert pred.needs_hitl is False


def test_verbal_tube_digits_path():
    pred = recognize_verbal(_empty_checkbox(), "tube_edta", backend="trocr")
    assert pred.field_id == "tube_edta"
    assert pred.raw_text == ""


def test_verbal_batch_independent_of_paddle():
    fields = recognize_fields({"others": _empty_checkbox(), "tube_edta": _empty_checkbox()})
    assert "others" in fields
    assert "tube_edta" in fields
    assert paddle_available() in {True, False}  # must not be required
    _ = trocr_available()  # stub OK in CI


def test_ingest_block1_zip(tmp_path: Path):
    zip_path = _write_mini_block1(tmp_path / "b1")
    base, is_temp = unzip_or_dir(zip_path)
    try:
        doc = load_block1_document(base / "docs" / "sample_sheet_01")
        assert doc.doc_id == "sample_sheet_01"
        assert "cbc" in doc.checkbox_crops
        assert "others" in doc.handwriting_crops
        assert doc.checkbox_crops["cbc"] is not None
    finally:
        if is_temp:
            import shutil

            shutil.rmtree(base, ignore_errors=True)


def test_kg_import_assume_without_qwen():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    ranked = kg.assume("others", "triglyc", {"ticked_ids": ["profile_lipid"]}, top_k=3)
    assert ranked
    assert ranked[0].canonical_id == "triglycerides"


def test_process_from_block1_nonverbal_only(tmp_path: Path):
    zip_path = _write_mini_block1(tmp_path / "b1")
    out = process_from_block1(
        zip_path,
        output_dir=tmp_path / "out_nv",
        kg=None,
        backend="lexicon",
        mode="nonverbal",
    )
    assert out["manifest"]["total_documents"] == 1
    hyp = json.loads((tmp_path / "out_nv" / "docs" / "sample_sheet_01" / "hypotheses.json").read_text())
    assert hyp["nonverbal"]["cbc"]["is_marked"] is True
    assert hyp["nonverbal"]["alt"]["is_marked"] is False
    assert hyp["verbal"] == {}
    assert hyp["ticked_test_ids"] == ["cbc"]


def test_process_from_block1_verbal_only(tmp_path: Path):
    zip_path = _write_mini_block1(tmp_path / "b1")
    kg = KnowledgeGraph.load(DEFAULT_KG)
    out = process_from_block1(
        zip_path,
        output_dir=tmp_path / "out_vb",
        kg=kg,
        backend="lexicon",
        mode="verbal",
    )
    hyp = json.loads((tmp_path / "out_vb" / "docs" / "sample_sheet_01" / "hypotheses.json").read_text())
    assert hyp["nonverbal"] == {}
    assert "others" in hyp["verbal"]
    assert "tube_edta" in hyp["verbal"]
    assert hyp["verbal"]["others"]["source"] == "empty"


def test_verbal_ink_without_trocr_is_hitl():
    pred = recognize_verbal(_ticked_checkbox(), "others", backend="trocr")
    if not trocr_available():
        assert pred.source in {"unavailable", "ink-present", "trocr"}
        if pred.source != "trocr":
            assert pred.needs_hitl is True


def test_process_from_block1_does_not_invent_tube_priors(tmp_path: Path):
    zip_path = _write_mini_block1(tmp_path / "b1")
    kg = KnowledgeGraph.load(DEFAULT_KG)
    out = process_from_block1(
        zip_path,
        output_dir=tmp_path / "out_no_prior",
        kg=kg,
        backend="lexicon",
        mode="both",
    )
    hyp = json.loads((tmp_path / "out_no_prior" / "docs" / "sample_sheet_01" / "hypotheses.json").read_text())
    assert hyp["verbal"]["tube_edta"]["source"] != "prior_expected"
    assert hyp["implied_tests"] == []


def test_process_from_block1_together_emits_hypotheses(tmp_path: Path):
    zip_path = _write_mini_block1(tmp_path / "b1")
    kg = KnowledgeGraph.load(DEFAULT_KG)
    zip_out = tmp_path / "block3_predictions_batch.zip"
    out = process_from_block1(
        zip_path,
        output_zip=zip_out,
        kg=kg,
        backend="lexicon",
        mode="both",
    )
    assert zip_out.exists()
    assert out["manifest"]["block"] == "block3"
    with zipfile.ZipFile(zip_out) as zf:
        names = zf.namelist()
        assert "manifest.json" in names
        assert "docs/sample_sheet_01/hypotheses.json" in names
        hyp = json.loads(zf.read("docs/sample_sheet_01/hypotheses.json"))
        assert hyp["nonverbal"]["cbc"]["is_marked"] is True
        assert "others" in hyp["verbal"]
        assert hyp["overall_confidence"] >= 0.0
        assert hyp["verbal"]["tube_edta"]["source"] != "prior_expected"

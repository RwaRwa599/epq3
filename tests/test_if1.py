"""if1 experimental pipeline: layout warp + coverage scorer (original Blocks 1–5 untouched)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from med_doc.htr.schemas import DocumentHypotheses, MarkPrediction
from med_doc.if1.if1block2 import estimate_groups_from_counts, load_groups, score
from med_doc.if1.if1block4 import process_from_if1block3
from med_doc.if1.layout_warp import layout_warp
from med_doc.if1.scorer import split_tiers
from med_doc.template import load_template


def _mark(fid: str, *, marked: bool, hitl: bool = False) -> MarkPrediction:
    return MarkPrediction(
        field_id=fid,
        is_marked=marked,
        confidence=0.9,
        needs_hitl=hitl,
        source="geometry",
    )


def test_groups_seeded_from_profile_bundles():
    groups = load_groups()
    ids = {g["id"] for g in groups}
    assert "profile_lipid" in ids
    lipid = next(g for g in groups if g["id"] == "profile_lipid")
    assert lipid["members"]["hdl"] == 1.0
    assert lipid["prior"] == 1.0


def test_scorer_profile_members_high_missing_low_extra_low():
    groups = load_groups()
    # Three of four lipid members + an unrelated tumour marker.
    initial = ["chol_total", "hdl", "ldl", "cea"]
    scored = score(initial, groups=groups)
    tiers = split_tiers(initial, scored)
    assert "hdl" in tiers["ordered_tests_high"]
    assert "chol_total" in tiers["ordered_tests_high"]
    assert "triglycerides" in tiers["low_missing"]
    assert "triglycerides" in tiers["ordered_tests_low"]
    assert "cea" in tiers["low_extra"]
    assert "cea" in tiers["ordered_tests_low"]
    assert "cea" not in tiers["ordered_tests_high"]


def test_estimate_groups_from_counts_is_placeholder():
    with pytest.raises(NotImplementedError, match="research placeholder"):
        estimate_groups_from_counts([])


def test_layout_warp_padded_header_records_method():
    from tests.test_normalization import _page_with_patient_header, render_canonical_form

    template = load_template()
    page = render_canonical_form(template)
    shifted, _pad = _page_with_patient_header(page, pad_frac=0.18)
    warped, meta = layout_warp(shifted, template)
    assert warped.shape[1] == template.width
    assert warped.shape[0] == template.height
    assert meta.get("method") in {"bar-gutter-affine", "page-quad", "min-area-rect", "full-frame"}
    if meta.get("method") != "bar-gutter-affine":
        assert meta.get("layout_fallback")


def test_if1block4_preserves_initial_ticked(tmp_path: Path):
    doc = tmp_path / "in" / "docs" / "synthetic"
    doc.mkdir(parents=True)
    hyp = DocumentHypotheses(
        doc_id="synthetic",
        nonverbal={
            "hdl": _mark("hdl", marked=True),
            "ldl": _mark("ldl", marked=True),
            "chol_total": _mark("chol_total", marked=True),
            "cea": _mark("cea", marked=True),
        },
        ticked_test_ids=["hdl", "ldl", "chol_total", "cea"],
        initial_ticked_test_ids=["hdl", "ldl", "chol_total", "cea"],
    )
    (doc / "hypotheses.json").write_text(hyp.model_dump_json(indent=2), encoding="utf-8")
    canvas = np.full((64, 64, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(doc / "canonical.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    (tmp_path / "in" / "manifest.json").write_text('{"block":"if1block3"}', encoding="utf-8")

    out = process_from_if1block3(tmp_path / "in", output_dir=tmp_path / "out")
    from med_doc.htr.schemas import DocumentPrediction

    prediction = DocumentPrediction.model_validate_json(
        (tmp_path / "out" / "docs" / "synthetic" / "prediction.json").read_text()
    )
    hyp_after = DocumentHypotheses.model_validate_json(
        (tmp_path / "out" / "docs" / "synthetic" / "hypotheses.json").read_text()
    )
    assert set(hyp_after.initial_ticked_test_ids) == {"cea", "chol_total", "hdl", "ldl"}
    assert set(prediction.initial_ticked_test_ids) == {"cea", "chol_total", "hdl", "ldl"}
    assert "triglycerides" in prediction.ordered_tests_low
    assert "cea" in prediction.ordered_tests_low
    assert "hdl" in prediction.ordered_tests_high
    assert out["manifest"]["block"] == "if1block4"


def test_if1block1_zip_feeds_original_block3(tmp_path: Path):
    from med_doc.htr.batch import process_from_block1
    from med_doc.if1.if1block1 import normalize_batch
    from tests.test_normalization import render_canonical_form

    template = load_template()
    page = render_canonical_form(template)
    img_path = tmp_path / "page.png"
    cv2.imwrite(str(img_path), cv2.cvtColor(page, cv2.COLOR_RGB2BGR))
    b1 = normalize_batch(
        [img_path],
        output_dir=tmp_path / "b1",
        output_zip=tmp_path / "if1block1.zip",
    )
    assert b1["manifest"]["block"] == "if1block1"
    b3 = process_from_block1(
        tmp_path / "if1block1.zip",
        output_dir=tmp_path / "b3",
        mode="nonverbal",
        mark_backend="geometry",
        vision_backend="off",
    )
    assert b3["manifest"]["total_documents"] == 1


def test_run_if1_is_live_pipeline_flag(tmp_path: Path):
    from med_doc.if1 import run_if1
    from med_doc.paths import SYNTHETIC_DIR
    from med_doc.pipeline import run_blocks_1_to_5

    blank = SYNTHETIC_DIR / "lab_request_v0_blank.png"
    img = tmp_path / "form.png"
    import shutil

    shutil.copy2(blank, img)
    pipe = run_if1(img, output_dir=tmp_path / "if1", output_mode="user")
    assert pipe["if1"] is True
    assert "if1" in run_blocks_1_to_5.__code__.co_varnames
    bundle = json.loads(Path(pipe["output_json"]).read_text(encoding="utf-8"))
    assert bundle["block"] == "block5"
    assert "ordered_tests_high" in bundle["orders"][0]


def test_lab_order_copies_if1_tiers():
    from med_doc.htr.schemas import DocumentPrediction
    from med_doc.review.lis import order_from_prediction

    pred = DocumentPrediction(
        doc_id="x",
        ticked_test_ids=["hdl"],
        ordered_tests_high=["hdl"],
        ordered_tests_low=["cea"],
        initial_ticked_test_ids=["hdl", "cea"],
        observed_tubes={"EDTA": 1, "SST": None, "NaF": None, "Citrate": None},
    )
    order = order_from_prediction(pred)
    assert order.ordered_tests_high == ["hdl"]
    assert order.ordered_tests_low == ["cea"]
    assert order.initial_ticked_test_ids == ["hdl", "cea"]

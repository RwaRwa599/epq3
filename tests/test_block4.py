"""Block 4: KG rescoring of synthetic Block 3 draft JSON (no OCR)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from med_doc.htr.fusion import fuse_handwriting
from med_doc.htr.schemas import DocumentHypotheses, HandwritingPrediction, MarkPrediction, VisionDraft
from med_doc.kg import KnowledgeGraph
from med_doc.paths import DEFAULT_KG
from med_doc.rescoring import (
    process_from_block3,
    rank_vision_ticks,
    rescore_hypotheses,
    trusted_tick_ids,
)
from med_doc.rescoring.combinations import CombinationFlag, ScriptedCombinationCritic, select_flags

GOLD10 = [
    "alt",
    "ana",
    "ferritin",
    "hs_crp",
    "profile_hepatitis",
    "profile_liver",
    "profile_thyroid",
    "troponin_i",
    "vitamin_b12",
    "vitamin_d",
]


def _mark(fid: str, *, marked: bool, hitl: bool = False, conf: float = 0.95) -> MarkPrediction:
    return MarkPrediction(
        field_id=fid,
        is_marked=marked,
        confidence=conf,
        ink_density=0.2 if marked else 0.0,
        needs_hitl=hitl,
        source="density",
    )


def _empty_hw(fid: str) -> HandwritingPrediction:
    return HandwritingPrediction(
        field_id=fid,
        raw_text="",
        canonical_value=None,
        canonical_id=None,
        confidence=0.9,
        source="empty",
        needs_hitl=False,
        hypotheses=[{"value": "", "score": 0.9, "source": "empty"}],
    )


def _digit_hw(fid: str, text: str) -> HandwritingPrediction:
    return HandwritingPrediction(
        field_id=fid,
        raw_text=text,
        canonical_value=text,
        canonical_id=fid,
        confidence=0.92,
        source="digits",
        needs_hitl=False,
        hypotheses=[{"value": text, "score": 0.92, "source": "digits"}],
    )


def _others_hw(text: str, *alts: str) -> HandwritingPrediction:
    hyps = [{"value": text, "score": 0.6, "source": "charset"}]
    for alt in alts:
        hyps.append({"value": alt, "score": 0.55, "source": "lexicon"})
    return HandwritingPrediction(
        field_id="others",
        raw_text=text,
        canonical_value=text,
        canonical_id=None,
        confidence=0.6,
        source="charset",
        needs_hitl=True,
        hypotheses=hyps,
    )


def _hyp(
    *,
    ticks: list[str],
    uncertain: list[str] | None = None,
    tubes: dict[str, str] | None = None,
    others: HandwritingPrediction | None = None,
    extra_marks: dict[str, MarkPrediction] | None = None,
    vision: VisionDraft | None = None,
) -> DocumentHypotheses:
    nonverbal = {fid: _mark(fid, marked=True) for fid in ticks}
    for fid in uncertain or []:
        nonverbal[fid] = _mark(fid, marked=True, hitl=True, conf=0.4)
    if extra_marks:
        nonverbal.update(extra_marks)
    verbal = {
        "tube_edta": _empty_hw("tube_edta"),
        "tube_cb": _empty_hw("tube_cb"),
        "tube_fl": _empty_hw("tube_fl"),
        "others": others or _empty_hw("others"),
        "received_at": _empty_hw("received_at"),
        "office_other": _empty_hw("office_other"),
    }
    for fid, text in (tubes or {}).items():
        verbal[fid] = _digit_hw(fid, text)
    return DocumentHypotheses(
        doc_id="synthetic",
        nonverbal=nonverbal,
        verbal=verbal,
        ticked_test_ids=list(nonverbal),
        implied_tests=[],
        overall_confidence=0.9,
        hitl_fields=[fid for fid, m in nonverbal.items() if m.needs_hitl],
        vision=vision or VisionDraft(),
    )


def test_trusted_ticks_exclude_hitl():
    hyp = _hyp(ticks=["alt"], uncertain=["cbc"])
    assert trusted_tick_ids(hyp.nonverbal) == ["alt"]


def test_ten_gold_ticks_empty_tubes_do_not_invent_counts():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    pred = rescore_hypotheses(_hyp(ticks=GOLD10), kg)
    expected = kg.calculate_expected_tubes(GOLD10)
    assert pred.ticked_test_ids == sorted(GOLD10)
    assert pred.expected_tubes == expected
    assert pred.observed_tubes.get("CB") is None
    assert pred.observed_tubes.get("EDTA") is None
    assert set(pred.observed_tubes) >= {"EDTA", "CB", "Fl", "Cit", "Urine", "Stool"}
    for field in pred.handwriting_fields.values():
        if field.field_id.startswith("tube_"):
            assert field.canonical_value is None
            assert field.source != "prior_expected"
    assert any("Missing tube observation" in d for d in pred.discrepancies)
    assert any(fid.startswith("tube_") for fid in pred.hitl_fields)


def test_false_positive_hitl_tick_does_not_add_tubes():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    pred = rescore_hypotheses(_hyp(ticks=GOLD10, uncertain=["cbc"]), kg)
    assert "cbc" not in pred.ticked_test_ids
    assert pred.expected_tubes == kg.calculate_expected_tubes(GOLD10)
    assert "EDTA" not in pred.expected_tubes
    assert "cbc" in pred.hitl_fields


def test_others_charset_soup_does_not_enter_lis():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    soup = "D 0 KA K LV T K A 0 N Y 7 X"
    hyp = _hyp(
        ticks=["profile_lipid"],
        others=HandwritingPrediction(
            field_id="others",
            raw_text=soup,
            canonical_value=soup,
            confidence=0.4,
            source="charset",
            needs_hitl=True,
            hypotheses=[{"value": soup, "score": 0.4, "source": "charset"}],
        ),
    )
    pred = rescore_hypotheses(hyp, kg)
    others = pred.handwriting_fields["others"]
    assert others.canonical_id is None
    assert others.source == "garbage"
    assert "triglycerides" not in pred.ticked_test_ids
    assert others.needs_hitl is True


def test_others_triglyc_with_lipid_profile():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(
        ticks=["profile_lipid"],
        others=_others_hw("triglyc", "triglycerides"),
    )
    pred = rescore_hypotheses(hyp, kg)
    others = pred.handwriting_fields["others"]
    assert others.canonical_id == "triglycerides"
    assert "triglycerides" in pred.ticked_test_ids
    assert "profile_lipid" in pred.ticked_test_ids


def test_tube_one_matches_expected_no_hitl():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(ticks=["cbc"], tubes={"tube_edta": "1"})
    pred = rescore_hypotheses(hyp, kg)
    assert pred.expected_tubes.get("EDTA") == 1
    assert pred.observed_tubes.get("EDTA") == 1
    assert pred.handwriting_fields["tube_edta"].needs_hitl is False
    assert "tube_edta" not in pred.hitl_fields
    assert not any("Tube" in d and "EDTA" in d for d in pred.discrepancies)


def test_tube_two_vs_expected_one_discrepancy():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(ticks=["cbc"], tubes={"tube_edta": "2"})
    pred = rescore_hypotheses(hyp, kg)
    assert pred.observed_tubes.get("EDTA") == 2
    assert pred.expected_tubes.get("EDTA") == 1
    assert pred.handwriting_fields["tube_edta"].needs_hitl is True
    assert "tube_edta" in pred.hitl_fields
    assert pred.discrepancies
    assert pred.is_valid is False


def test_implausible_tube_ocr_is_dropped():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(ticks=["cbc"], tubes={"tube_edta": "11"})
    pred = rescore_hypotheses(hyp, kg)
    assert pred.observed_tubes.get("EDTA") is None
    assert pred.handwriting_fields["tube_edta"].needs_hitl is True
    assert any("Implausible tube count" in w for w in pred.warnings)


def test_empty_tube_expected_one_hitl_observed_null():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    pred = rescore_hypotheses(_hyp(ticks=["cbc"]), kg)
    edta = pred.handwriting_fields["tube_edta"]
    assert edta.canonical_value is None
    assert edta.source == "empty"
    assert edta.source != "prior_expected"
    assert pred.observed_tubes.get("EDTA") is None
    assert pred.expected_tubes.get("EDTA") == 1
    assert edta.needs_hitl is True
    assert "tube_edta" in pred.hitl_fields
    assert any("Missing tube observation" in d for d in pred.discrepancies)


def test_fuse_empty_tube_never_prior_expected():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    fused = fuse_handwriting(
        "tube_edta",
        "",
        0.9,
        "empty",
        kg=kg,
        ticked_ids=["cbc"],
    )
    assert fused.canonical_value is None
    assert fused.source == "empty"
    assert fused.needs_hitl is True


def test_unparsed_received_at_is_hitl():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(ticks=["cbc"])
    hyp.verbal["received_at"] = HandwritingPrediction(
        field_id="received_at",
        raw_text="someday",
        canonical_value="someday",
        confidence=0.9,
        source="charset",
        needs_hitl=False,
        hypotheses=[{"value": "someday", "score": 0.9, "source": "charset"}],
    )
    pred = rescore_hypotheses(hyp, kg)
    rec = pred.handwriting_fields["received_at"]
    assert rec.needs_hitl is True
    assert rec.source != "date_parse"
    assert rec.canonical_id is None


def test_process_from_block3_writes_prediction_keeps_hypotheses(tmp_path: Path):
    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(ticks=["cbc"], tubes={"tube_edta": "1"})
    src = tmp_path / "b3" / "docs" / "synthetic"
    src.mkdir(parents=True)
    (src / "hypotheses.json").write_text(hyp.model_dump_json(indent=2), encoding="utf-8")
    original = (src / "hypotheses.json").read_text(encoding="utf-8")
    zip_out = tmp_path / "block4.zip"
    out = process_from_block3(tmp_path / "b3", output_dir=tmp_path / "b4", output_zip=zip_out, kg=kg)
    assert out["manifest"]["block"] == "block4"
    pred_path = tmp_path / "b4" / "docs" / "synthetic" / "prediction.json"
    hyp_path = tmp_path / "b4" / "docs" / "synthetic" / "hypotheses.json"
    assert pred_path.exists()
    assert json.loads(hyp_path.read_text())["verbal"]["tube_edta"]["source"] != "prior_expected"
    assert hyp_path.read_text(encoding="utf-8") == original
    payload = json.loads(pred_path.read_text())
    assert payload["ticked_test_ids"] == ["cbc"]
    assert payload["observed_tubes"]["EDTA"] == 1
    with zipfile.ZipFile(zip_out) as zf:
        assert "docs/synthetic/prediction.json" in zf.namelist()
        assert "docs/synthetic/hypotheses.json" in zf.namelist()


def test_vision_only_tick_is_hitl_not_lis():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(
        ticks=["cbc"],
        tubes={"tube_edta": "1"},
        vision=VisionDraft(
            source="vision",
            ticked_field_ids=["cbc", "ca125"],
            handwriting={},
        ),
    )
    pred = rescore_hypotheses(hyp, kg)
    assert "ca125" not in pred.ticked_test_ids
    assert "ca125" in pred.hitl_fields
    assert any("Vision tick" in w for w in pred.warnings)
    ranked = rank_vision_ticks(["cbc"], ["ca125"], hyp.verbal, kg)
    assert ranked[0]["field_id"] == "ca125"
    assert "ca125" not in pred.implied_tests or "ca125" not in pred.ticked_test_ids


def test_rank_vision_ticks_orders_hitl_never_unions_lis():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(
        ticks=["profile_lipid"],
        vision=VisionDraft(
            source="vision",
            ticked_field_ids=["profile_lipid", "triglycerides", "ca125"],
            handwriting={},
        ),
    )
    pred = rescore_hypotheses(hyp, kg)
    assert "triglycerides" not in pred.ticked_test_ids
    assert "ca125" not in pred.ticked_test_ids
    assert pred.hitl_fields.index("triglycerides") < pred.hitl_fields.index("ca125")
    assert any("kg_score=" in w and "triglycerides" in w for w in pred.warnings)


def test_vision_others_fills_empty_charset_and_kg_ranks():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(
        ticks=["profile_lipid"],
        others=_empty_hw("others"),
        vision=VisionDraft(
            source="vision",
            ticked_field_ids=["profile_lipid"],
            handwriting={"others": "triglyc"},
        ),
    )
    pred = rescore_hypotheses(hyp, kg)
    others = pred.handwriting_fields["others"]
    assert others.source == "vision+kg" or others.canonical_id == "triglycerides"
    assert others.canonical_id == "triglycerides"
    assert "triglycerides" in pred.ticked_test_ids


def test_select_flags_threshold_and_cap():
    flags = [CombinationFlag(f"f{i}", "missing_likely", 0.9) for i in range(20)]
    flags.append(CombinationFlag("low", "missing_likely", 0.1))
    kept = select_flags(flags, threshold=0.55, cap=16)
    assert len(kept) == 16
    assert all(f.confidence >= 0.55 for f in kept)
    assert "low" not in {f.field_id for f in kept}


def test_low_confidence_combination_is_not_reviewed():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(ticks=["cbc"])
    critic = ScriptedCombinationCritic(
        [CombinationFlag("hdl", "missing_likely", 0.2, reason="weak")]
    )
    pred = rescore_hypotheses(hyp, kg, critic=critic, combo_threshold=0.55)
    assert "hdl" not in pred.hitl_fields
    assert "hdl" not in pred.ticked_test_ids


def test_high_confidence_missing_likely_is_hitl_not_lis_without_3a():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(ticks=["profile_lipid"])
    critic = ScriptedCombinationCritic(
        [CombinationFlag("hdl", "missing_likely", 0.91, reason="correlates_with_lipid")]
    )
    pred = rescore_hypotheses(hyp, kg, critic=critic)
    assert "hdl" in pred.hitl_fields
    assert "hdl" not in pred.ticked_test_ids
    assert any("Combination missing_likely: hdl" in w for w in pred.warnings)


def test_second_3a_pass_can_commit_missing_likely_slash():
    from test_block3_verbal_nonverbal import _ticked_checkbox

    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(ticks=["profile_lipid"])
    critic = ScriptedCombinationCritic(
        [CombinationFlag("triglycerides", "missing_likely", 0.9, reason="lipid_panel")]
    )
    pred = rescore_hypotheses(
        hyp,
        kg,
        critic=critic,
        checkbox_crops={"triglycerides": _ticked_checkbox(48)},
    )
    if "triglycerides" in pred.ticked_test_ids:
        mark = pred.checkbox_marks["triglycerides"]
        assert mark.is_marked
        assert mark.source != "llm"
    else:
        assert "triglycerides" in pred.hitl_fields


def test_odd_member_empty_crop_leaves_review_or_drops():
    from test_block3_verbal_nonverbal import _empty_checkbox

    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(ticks=["cbc", "ca125"])
    critic = ScriptedCombinationCritic(
        [CombinationFlag("ca125", "odd_member", 0.88, reason="rare_with_cbc")]
    )
    pred = rescore_hypotheses(
        hyp,
        kg,
        critic=critic,
        checkbox_crops={"ca125": _empty_checkbox(48)},
    )
    assert any("odd_member" in w for w in pred.warnings)
    mark = pred.checkbox_marks.get("ca125")
    if mark is not None and not mark.is_marked:
        assert "ca125" not in pred.ticked_test_ids
    else:
        assert "ca125" in pred.hitl_fields

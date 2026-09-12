"""Block 4: KG rescoring of synthetic Block 3 draft JSON (no OCR)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from med_doc.htr.fusion import fuse_handwriting
from med_doc.htr.schemas import DocumentHypotheses, HandwritingPrediction, MarkPrediction
from med_doc.kg import KnowledgeGraph
from med_doc.paths import DEFAULT_KG
from med_doc.rescoring import process_from_block3, rescore_hypotheses, trusted_tick_ids

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

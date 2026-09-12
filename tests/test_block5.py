"""Block 5: HiTL queue, review patches, LIS order (synthetic drafts, no OCR)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from med_doc.kg import KnowledgeGraph
from med_doc.paths import DEFAULT_KG
from med_doc.rescoring import rescore_hypotheses
from med_doc.review import (
    ReviewPatch,
    ScriptedLlmRanker,
    apply_patches,
    attach_llm_suggestions,
    build_hitl_queue,
    commit_hypotheses,
    order_from_prediction,
    process_from_block4,
)
from test_block4 import _hyp, _others_hw


def test_clean_prediction_auto_passes_to_order():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(ticks=["cbc"], tubes={"tube_edta": "1"})
    pred = rescore_hypotheses(hyp, kg)
    queue = build_hitl_queue(pred, hyp)
    assert queue == []
    order = order_from_prediction(pred)
    assert order.needs_review is False
    assert "cbc" in order.ordered_tests
    assert order.observed_tubes.get("EDTA") == 1
    assert set(order.ordered_tests) == set(pred.ticked_test_ids) | set(pred.implied_tests)


def test_uncertain_tick_queued_until_confirm():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(ticks=["alt"], uncertain=["cbc"])
    pred = rescore_hypotheses(hyp, kg)
    queue = build_hitl_queue(pred, hyp)
    assert any(item.field_id == "cbc" and item.kind == "tick" for item in queue)
    assert "cbc" not in pred.ticked_test_ids

    patched, committed = commit_hypotheses(
        hyp, [ReviewPatch(field_id="cbc", action="confirm_tick")], kg
    )
    assert patched.nonverbal["cbc"].needs_hitl is False
    assert patched.nonverbal["cbc"].is_marked is True
    assert "cbc" in committed.ticked_test_ids
    assert committed.expected_tubes.get("EDTA") == 1


def test_reject_uncertain_tick_keeps_it_out():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(ticks=["alt"], uncertain=["cbc"])
    _, committed = commit_hypotheses(
        hyp, [ReviewPatch(field_id="cbc", action="reject_tick")], kg
    )
    assert "cbc" not in committed.ticked_test_ids
    assert "EDTA" not in committed.expected_tubes


def test_nurse_override_fills_missing_tube():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(ticks=["cbc"])
    pred = rescore_hypotheses(hyp, kg)
    assert pred.observed_tubes.get("EDTA") is None
    assert hyp.verbal["tube_edta"].raw_text == ""

    patched, committed = commit_hypotheses(
        hyp,
        [ReviewPatch(field_id="tube_edta", action="set_tube", value="1")],
        kg,
    )
    assert patched.verbal["tube_edta"].source == "hitl"
    assert hyp.verbal["tube_edta"].raw_text == ""
    assert committed.observed_tubes.get("EDTA") == 1
    assert not any("Missing tube observation" in d for d in committed.discrepancies)
    assert "tube_edta" not in committed.hitl_fields
    order = order_from_prediction(committed)
    assert order.needs_review is False
    assert order.is_valid is True


def test_llm_suggestion_rejected_when_not_in_nbest():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(ticks=["profile_lipid"], others=_others_hw("zzzz"))
    pred = rescore_hypotheses(hyp, kg)
    queue = build_hitl_queue(pred, hyp)
    ranker = ScriptedLlmRanker({"others": "triglycerides"})
    attached = attach_llm_suggestions(queue, ranker=ranker, enabled=True, kg=kg)
    others_item = next(i for i in attached if i.field_id == "others")
    assert others_item.llm_suggestions == []


def test_llm_suggestion_kept_when_in_nbest_but_not_auto_applied():
    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(ticks=["profile_lipid"], others=_others_hw("triglyc", "triglycerides"))
    pred = rescore_hypotheses(hyp, kg)
    queue = build_hitl_queue(pred, hyp)
    ranker = ScriptedLlmRanker({"others": "triglyc"})
    attached = attach_llm_suggestions(queue, ranker=ranker, enabled=True, kg=kg)
    others_item = next((i for i in attached if i.field_id == "others"), None)
    if others_item is not None:
        assert others_item.llm_suggestions
        assert others_item.llm_suggestions[0]["value"] == "triglyc"
    # LLM must not mutate drafts
    assert hyp.verbal["others"].source != "llm"


def test_apply_patches_does_not_mutate_original():
    hyp = _hyp(ticks=["cbc"])
    apply_patches(hyp, [ReviewPatch(field_id="tube_edta", action="set_tube", value="2")])
    assert hyp.verbal["tube_edta"].raw_text == ""


def test_process_from_block4_writes_order_keeps_hypotheses(tmp_path: Path):
    kg = KnowledgeGraph.load(DEFAULT_KG)
    hyp = _hyp(ticks=["cbc"])
    src = tmp_path / "b4" / "docs" / "synthetic"
    src.mkdir(parents=True)
    (src / "hypotheses.json").write_text(hyp.model_dump_json(indent=2), encoding="utf-8")
    pred = rescore_hypotheses(hyp, kg)
    (src / "prediction.json").write_text(pred.model_dump_json(indent=2), encoding="utf-8")
    original = (src / "hypotheses.json").read_text(encoding="utf-8")

    zip_out = tmp_path / "block5.zip"
    out = process_from_block4(
        tmp_path / "b4",
        output_dir=tmp_path / "b5",
        output_zip=zip_out,
        kg=kg,
        reviews={"synthetic": [ReviewPatch(field_id="tube_edta", action="set_tube", value="1")]},
    )
    assert out["manifest"]["block"] == "block5"
    hyp_path = tmp_path / "b5" / "docs" / "synthetic" / "hypotheses.json"
    assert hyp_path.read_text(encoding="utf-8") == original
    order = json.loads((tmp_path / "b5" / "docs" / "synthetic" / "order.json").read_text())
    assert order["observed_tubes"]["EDTA"] == 1
    assert order["needs_review"] is False
    review = json.loads((tmp_path / "b5" / "docs" / "synthetic" / "review.json").read_text())
    assert review["patches"][0]["action"] == "set_tube"
    with zipfile.ZipFile(zip_out) as zf:
        names = zf.namelist()
        assert "docs/synthetic/order.json" in names
        assert "docs/synthetic/hypotheses.json" in names
        assert "docs/synthetic/prediction.json" in names
        assert "docs/synthetic/prediction.committed.json" in names

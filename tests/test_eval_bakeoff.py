"""Local VL / Instruct bakeoff against gold (no PHI images in git)."""

from __future__ import annotations

import json
from pathlib import Path

from med_doc.eval.bakeoff import text_bakeoff, vision_bakeoff
from med_doc.eval.__main__ import main
from med_doc.htr.schemas import DocumentHypotheses, HandwritingPrediction, VisionDraft
from med_doc.htr.vision import ScriptedVisionClient
from med_doc.review.llm import ScriptedLlmRanker
from test_block3_verbal_nonverbal import _write_mini_block1


def test_vision_bakeoff_vs_gold_and_3a_not_vlm_as_truth(tmp_path: Path):
    root = _write_mini_block1(tmp_path / "b1")
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            {
                "sheets": [
                    {
                        "doc_id": "sample_sheet_01",
                        "ticked_field_ids": ["cbc"],
                        "others": "AFP",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def factory(model: str) -> ScriptedVisionClient:
        return ScriptedVisionClient(
            {"ticked_field_ids": ["cbc"], "handwriting": {"others": "AFP"}},
            model=model,
        )

    result = vision_bakeoff(root, gold, ["qwen2.5vl:7b"], client_factory=factory)
    assert result["truth"] == "gold_and_3a_never_vlm"
    row = result["rows"][0]
    assert row["ticks_vs_gold"]["recall"] == 1.0
    assert row["others"]["exact"] is True


def test_text_bakeoff_instruct_does_not_invent_ticks(tmp_path: Path):
    hyp = DocumentHypotheses(
        doc_id="sample_sheet_01",
        verbal={
            "others": HandwritingPrediction(
                field_id="others",
                raw_text="triglyc",
                hypotheses=[
                    {"value": "triglyc", "score": 0.6},
                    {"value": "triglycerides", "score": 0.5},
                ],
            )
        },
        vision=VisionDraft(source="vision", handwriting={"others": "triglyc"}),
    )
    hyp_path = tmp_path / "hypotheses.json"
    hyp_path.write_text(hyp.model_dump_json(), encoding="utf-8")
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps({"sheets": [{"doc_id": "sample_sheet_01", "others": "triglycerides"}]}),
        encoding="utf-8",
    )
    result = text_bakeoff(
        hyp_path,
        gold,
        ["qwen2.5:7b-instruct"],
        ranker_factory=lambda m: ScriptedLlmRanker({"others": "triglycerides"}),
    )
    row = result["rows"][0]
    assert row["tick_suggestions"] == []
    assert row["suggestion"] == "triglycerides"


def test_eval_cli_lists_bakeoff_commands():
    try:
        main(["vision-bakeoff", "-h"])
    except SystemExit as exc:
        assert exc.code == 0

"""Block 3c local VLM drafts: whitelist + off-by-default."""

from __future__ import annotations

import inspect

import numpy as np

from med_doc.htr.batch import process_from_block1
from med_doc.htr.ingest import Block1Document, load_block1_document
from med_doc.htr.schemas import DocumentHypotheses
from med_doc.htr.vision import (
    MAX_DISAGREEMENT_CROPS,
    ScriptedVisionClient,
    disagreement_crop_ids,
    parse_vision_json,
    run_vision_draft,
    whitelist_vision_payload,
)
from med_doc.pipeline import run_blocks_1_to_5


def test_parse_vision_json_strips_fences():
    raw = 'Sure.\n```json\n{"ticked_field_ids":["cbc"],"handwriting":{"others":"AFP"}}\n```\n'
    obj = parse_vision_json(raw)
    assert obj["ticked_field_ids"] == ["cbc"]


def test_whitelist_drops_unknown_ids_and_keeps_catalogue():
    ticks, hw = whitelist_vision_payload(
        {
            "ticked_field_ids": ["cbc", "not_a_test", "alt"],
            "handwriting": {"others": "AFP", "ssn": "nope", "tube_edta": "1"},
        },
        allowed_ticks={"cbc", "alt"},
        allowed_hw={"others", "tube_edta"},
    )
    assert ticks == ["cbc", "alt"]
    assert hw == {"others": "AFP", "tube_edta": "1"}


def test_scripted_client_returns_json():
    client = ScriptedVisionClient({"ticked_field_ids": ["cbc"], "handwriting": {}})
    assert "cbc" in client.complete("x", [])


def test_document_hypotheses_has_one_vision_field():
    names = [name for name in DocumentHypotheses.model_fields if name == "vision"]
    assert names == ["vision"]


def test_vision_model_is_on_pipeline_apis():
    assert "vision_model" in inspect.signature(process_from_block1).parameters
    assert "vision_model" in inspect.signature(run_blocks_1_to_5).parameters


def test_disagreement_crop_list_is_capped():
    crops = {f"f{i:02d}": np.zeros((8, 8, 3), dtype=np.uint8) for i in range(20)}
    doc = Block1Document(doc_id="cap", doc_dir=None, metadata={}, checkbox_crops=crops)  # type: ignore[arg-type]
    ids = disagreement_crop_ids(doc, set(), set(crops), cap=MAX_DISAGREEMENT_CROPS)
    assert len(ids) == MAX_DISAGREEMENT_CROPS
    assert MAX_DISAGREEMENT_CROPS == 16


def test_disagreement_pass_sends_only_mismatch_crops(tmp_path):
    from test_block3_verbal_nonverbal import _write_mini_block1

    root = _write_mini_block1(tmp_path / "b1")
    doc = load_block1_document(tmp_path / "b1" / "docs" / "sample_sheet_01")
    client = ScriptedVisionClient(
        {"ticked_field_ids": ["cbc", "alt"], "handwriting": {"others": "AFP"}},
        model="qwen2.5vl:7b",
    )
    draft = run_vision_draft(doc, client=client, enabled=True, geometry_ticks=["cbc"])
    assert draft.model == "qwen2.5vl:7b"
    assert len(client.calls) == 2
    assert len(client.calls[1][1]) == 1
    assert "disagreement pass" in draft.notes
    assert "alt" in draft.ticked_field_ids

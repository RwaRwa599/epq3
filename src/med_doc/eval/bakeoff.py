"""Local VL / Instruct bakeoff vs gitignored gold. Never treats the VLM as truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from med_doc.eval.mark_gold import GoldSet, load_gold
from med_doc.htr.ingest import iter_block1_documents, unzip_or_dir, cleanup_temp
from med_doc.htr.nonverbal import classify_marks
from med_doc.htr.schemas import DocumentHypotheses, HandwritingPrediction
from med_doc.htr.vision import VisionClient, run_vision_draft
from med_doc.kg.textutil import normalize_text, token_similarity
from med_doc.review.llm import LlmRanker, OllamaRanker
from med_doc.review.schemas import HitlItem


def _pr(pred: set[str], gold: set[str]) -> dict[str, float | int]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4), "recall": round(recall, 4)}


def _hw_scores(pred_text: str, gold_text: str) -> dict[str, Any]:
    a = (pred_text or "").strip()
    b = (gold_text or "").strip()
    exact = normalize_text(a) == normalize_text(b) if (a or b) else True
    return {
        "pred": a,
        "gold": b,
        "exact": exact,
        "token": round(token_similarity(a, b), 4) if (a or b) else 1.0,
    }


def vision_bakeoff(
    block1: str | Path,
    gold_path: str | Path,
    models: list[str],
    *,
    client_factory: Callable[[str], VisionClient] | None = None,
) -> dict[str, Any]:
    """VL models read crops. Tick PR vs 3a and vs gold; others vs gold when labeled."""
    gold = load_gold(gold_path)
    by_doc = gold.by_doc()
    base, is_temp = unzip_or_dir(block1)
    rows: list[dict[str, Any]] = []
    try:
        docs = list(iter_block1_documents(base))
        for model in models:
            for doc in docs:
                sheet = by_doc.get(doc.doc_id)
                gold_ticks = set(sheet.ticked_field_ids) if sheet else set()
                gold_others = sheet.others if sheet else ""
                cb_meta = (doc.fields.get("checkboxes") or {}) if doc.fields else {}
                merged = {
                    fid: {**(cb_meta.get(fid) or {}), **((doc.detected_marks or {}).get(fid) or {})}
                    for fid in doc.checkbox_crops
                }
                marks = classify_marks(doc.checkbox_crops, fallbacks=merged)
                geom = {fid for fid, m in marks.items() if m.is_marked}
                client = client_factory(model) if client_factory else None
                draft = run_vision_draft(
                    doc,
                    client=client,
                    enabled=True,
                    model=model,
                    geometry_ticks=sorted(geom),
                )
                vis = set(draft.ticked_field_ids)
                others = (draft.handwriting or {}).get("others") or ""
                rows.append(
                    {
                        "model": model,
                        "doc_id": doc.doc_id,
                        "n_images_note": draft.notes,
                        "ticks_vs_3a": _pr(vis, geom),
                        "ticks_vs_gold": _pr(vis, gold_ticks) if sheet else None,
                        "others": _hw_scores(others, gold_others) if gold_others else _hw_scores(others, others),
                        "vision_ticks": sorted(vis),
                        "geometry_ticks": sorted(geom),
                    }
                )
    finally:
        cleanup_temp(base, is_temp)
    return {"kind": "vision", "truth": "gold_and_3a_never_vlm", "rows": rows}


def text_bakeoff(
    hypotheses_source: str | Path,
    gold_path: str | Path,
    models: list[str],
    *,
    ranker_factory: Callable[[str], LlmRanker] | None = None,
) -> dict[str, Any]:
    """Instruct models reorder frozen 3b/3c n-best. They cannot add ticks."""
    gold = load_gold(gold_path)
    by_doc = gold.by_doc()
    path = Path(hypotheses_source)
    files: list[Path] = []
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("hypotheses.json"))
    rows: list[dict[str, Any]] = []
    for model in models:
        ranker = ranker_factory(model) if ranker_factory else OllamaRanker(model=model)
        for hyp_path in files:
            hyp = DocumentHypotheses.model_validate_json(hyp_path.read_text(encoding="utf-8"))
            sheet = by_doc.get(hyp.doc_id)
            others: HandwritingPrediction | None = hyp.verbal.get("others")
            nbest: list[str] = []
            if others is not None:
                if (others.raw_text or "").strip():
                    nbest.append(others.raw_text.strip())
                for row in others.hypotheses or []:
                    if isinstance(row, dict):
                        text = str(row.get("value") or "").strip()
                        if text and text not in nbest:
                            nbest.append(text)
                vis = (hyp.vision.handwriting or {}).get("others") if hyp.vision else ""
                if vis and vis not in nbest:
                    nbest.append(vis)
            item = HitlItem(field_id="others", kind="write_in", nbest=nbest, raw_text=nbest[0] if nbest else "")
            tick_item = HitlItem(field_id="cbc", kind="tick", nbest=["cbc"])
            suggestions = ranker.suggest(item, nbest=nbest)
            tick_suggestions = ranker.suggest(tick_item, nbest=["cbc"])
            gold_others = sheet.others if sheet else ""
            chosen = suggestions[0]["value"] if suggestions else (nbest[0] if nbest else "")
            rows.append(
                {
                    "model": model,
                    "doc_id": hyp.doc_id,
                    "nbest": nbest,
                    "suggestion": chosen,
                    "tick_suggestions": tick_suggestions,
                    "others": _hw_scores(chosen, gold_others) if gold_others else None,
                }
            )
    return {"kind": "text_instruct", "truth": "gold_never_vlm", "rows": rows}


def dump_bakeoff(result: dict[str, Any], dest: str | Path | None = None) -> str:
    blob = json.dumps(result, indent=2)
    if dest:
        Path(dest).write_text(blob, encoding="utf-8")
    return blob

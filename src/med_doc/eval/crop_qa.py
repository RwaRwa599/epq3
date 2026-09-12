"""Block 1 crop QA: is this checkbox PNG a printed square or a shifted label?"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from med_doc.eval.mark_gold import GoldSet, load_gold
from med_doc.htr.ingest import cleanup_temp, iter_block1_documents, unzip_or_dir
from med_doc.htr.marks import classify_mark, looks_like_text_line
from med_doc.normalization.block1c import has_hollow_ring


def diagnose_crop(crop, *, gold_ticked: bool, field_id: str) -> dict[str, Any]:
    pred = classify_mark(crop, field_id)
    hollow = bool(has_hollow_ring(crop)) if crop is not None and getattr(crop, "size", 0) else False
    textish = bool(looks_like_text_line(crop)) if crop is not None and getattr(crop, "size", 0) else False
    if not hollow:
        if gold_ticked:
            cause = "block1_shift"
        elif pred.is_marked:
            cause = "classifier_fp"
        else:
            cause = "ok"
    elif gold_ticked and not pred.is_marked:
        cause = "classifier_fn"
    elif (not gold_ticked) and pred.is_marked:
        cause = "classifier_fp"
    else:
        cause = "ok"
    if textish and gold_ticked:
        cause = "block1_shift"
    return {
        "field_id": field_id,
        "gold_ticked": gold_ticked,
        "pred_ticked": bool(pred.is_marked),
        "source": pred.source,
        "hollow_ring": hollow,
        "looks_like_text": textish,
        "cause": cause,
    }


def crop_qa(
    block1_source: str | Path,
    gold: GoldSet | str | Path,
    *,
    field_id: str | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(gold, GoldSet):
        gold = load_gold(gold)
    by_doc = gold.by_doc()
    base, is_temp = unzip_or_dir(block1_source)
    rows: list[dict[str, Any]] = []
    try:
        for doc in iter_block1_documents(base):
            sheet = by_doc.get(doc.doc_id)
            if sheet is None:
                continue
            ticked = set(sheet.ticked_field_ids)
            fids = [field_id] if field_id else list(doc.checkbox_crops)
            for fid in fids:
                crop = doc.checkbox_crops.get(fid)
                if crop is None:
                    continue
                row = diagnose_crop(crop, gold_ticked=fid in ticked, field_id=fid)
                row["doc_id"] = doc.doc_id
                rows.append(row)
    finally:
        cleanup_temp(base, is_temp)
    return rows

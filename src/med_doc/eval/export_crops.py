"""Split Block 1 checkbox PNGs into empty vs tick using gold field ids (local only)."""

from __future__ import annotations

from pathlib import Path

import cv2

from med_doc.eval.mark_gold import GoldSet, SheetGold, dump_gold, load_gold
from med_doc.htr.ingest import cleanup_temp, iter_block1_documents, unzip_or_dir


def export_labeled_crops(
    block1_source: str | Path,
    gold: GoldSet | str | Path,
    out_dir: str | Path,
) -> dict[str, int]:
    """Copy each checkbox PNG into ``out_dir/{doc_id}/empty|tick/{field_id}.png``.

    Gold is a list of ticked field ids per ``doc_id``. Unticked boxes go to ``empty``.
    Write under ``data/labels/crops`` (gitignored) — never commit clinic PNGs.
    """
    if not isinstance(gold, GoldSet):
        gold = load_gold(gold)
    by_doc = gold.by_doc()
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)

    base, is_temp = unzip_or_dir(block1_source)
    n_empty = n_tick = n_skip = 0
    try:
        for doc in iter_block1_documents(base):
            sheet = by_doc.get(doc.doc_id)
            if sheet is None:
                n_skip += 1
                continue
            ticked = set(sheet.ticked_field_ids)
            empty_dir = dest / doc.doc_id / "empty"
            tick_dir = dest / doc.doc_id / "tick"
            empty_dir.mkdir(parents=True, exist_ok=True)
            tick_dir.mkdir(parents=True, exist_ok=True)
            for fid, crop in doc.checkbox_crops.items():
                if crop is None or getattr(crop, "size", 0) == 0:
                    continue
                folder = tick_dir if fid in ticked else empty_dir
                bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(folder / f"{fid}.png"), bgr)
                if fid in ticked:
                    n_tick += 1
                else:
                    n_empty += 1
            dump_gold(
                SheetGold(doc_id=doc.doc_id, ticked_field_ids=sheet.ticked_field_ids, notes=sheet.notes),
                dest / doc.doc_id / "gold.json",
            )
    finally:
        cleanup_temp(base, is_temp)
    return {"empty": n_empty, "tick": n_tick, "unlabeled_docs": n_skip}

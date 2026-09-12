"""Build a HiTL work queue from Block 4 prediction + Block 3 drafts."""

from __future__ import annotations

from pathlib import Path

from typing import Literal

from med_doc.htr.schemas import DocumentHypotheses, DocumentPrediction, HandwritingPrediction, MarkPrediction
from med_doc.review.schemas import HitlItem

_DATE_FIELDS = {"received_at", "date", "sample_received"}
Kind = Literal["tick", "tube", "write_in", "date", "other"]


def _kind(field_id: str, hyp: DocumentHypotheses) -> Kind:
    if field_id.startswith("tube_"):
        return "tube"
    if field_id == "others":
        return "write_in"
    if field_id in _DATE_FIELDS:
        return "date"
    if field_id in hyp.nonverbal:
        return "tick"
    return "other"


def _nbest(field: HandwritingPrediction | None) -> list[str]:
    if field is None:
        return []
    out: list[str] = []
    if (field.raw_text or "").strip():
        out.append(field.raw_text.strip())
    for row in field.hypotheses or []:
        if not isinstance(row, dict):
            continue
        text = str(row.get("value") or row.get("text") or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _crop_path(doc_dir: Path | None, field_id: str) -> str | None:
    if doc_dir is None:
        return None
    cb = doc_dir / "crops" / "checkboxes" / f"{field_id}.png"
    hw = doc_dir / "crops" / "handwriting" / f"{field_id}.png"
    if cb.exists():
        return f"crops/checkboxes/{field_id}.png"
    if hw.exists():
        return f"crops/handwriting/{field_id}.png"
    return None


def _reason(field_id: str, pred: DocumentPrediction, kind: str) -> str:
    for line in pred.discrepancies:
        if field_id in line or (kind == "tube" and field_id.replace("tube_", "").upper() in line):
            return line
    if kind == "tick":
        return "Uncertain tick excluded from tube prior"
    if kind == "tube":
        return "Tube observation missing or mismatched"
    if kind == "write_in":
        return "Write-in not accepted as tier-1 catalogue id"
    if kind == "date":
        return "Unparsed received_at"
    return "Needs human review"


def build_hitl_queue(
    pred: DocumentPrediction,
    hyp: DocumentHypotheses,
    *,
    doc_dir: Path | None = None,
) -> list[HitlItem]:
    items: list[HitlItem] = []
    for fid in pred.hitl_fields:
        kind = _kind(fid, hyp)
        mark: MarkPrediction | None = hyp.nonverbal.get(fid) or pred.checkbox_marks.get(fid)
        draft: HandwritingPrediction | None = hyp.verbal.get(fid) or pred.handwriting_fields.get(fid)
        items.append(
            HitlItem(
                field_id=fid,
                kind=kind,
                reason=_reason(fid, pred, kind),
                crop_path=_crop_path(doc_dir, fid),
                raw_text=draft.raw_text if draft else "",
                nbest=_nbest(draft),
                is_marked=mark.is_marked if mark else None,
                canonical_id=draft.canonical_id if draft else None,
            )
        )
    return items

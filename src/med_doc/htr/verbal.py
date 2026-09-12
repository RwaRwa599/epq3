"""Verbal Block 3: field-typed handwriting drafts. Independent of PaddleOCR."""

from __future__ import annotations

import numpy as np

from med_doc.htr.fusion import fuse_handwriting
from med_doc.htr.recognizer import recognize_handwriting
from med_doc.htr.schemas import HandwritingPrediction
from med_doc.kg.graph import KnowledgeGraph

HITL_TAU = 0.75


def trocr_available() -> bool:
    """True when transformers + a TrOCR checkpoint can be loaded."""
    try:
        import torch  # noqa: F401
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # noqa: F401
    except Exception:
        return False
    return True


def recognize_verbal(
    crop: np.ndarray | None,
    field_id: str,
    *,
    backend: str = "trocr",
    blank: np.ndarray | None = None,
) -> HandwritingPrediction:
    """Transcribe one handwriting crop. Does not classify checkboxes.

    Tube fields keep digits only. Dates are grammar-normalized. `others` stays
    a raw n-best draft. KG `assume()` is Block 4.
    """
    draft = recognize_handwriting(crop, field_id, backend=backend, blank=blank)
    empty = not (draft.text or "").strip()
    engine_gap = draft.source in {"unavailable", "ink-present", "trocr-nodigit", "digit-reject"}
    needs_hitl = engine_gap or (not empty and draft.confidence < HITL_TAU)
    if empty and draft.source == "empty":
        needs_hitl = False
    canonical = draft.text or None
    canonical_id = None
    if field_id.startswith("tube_") and draft.text:
        canonical_id = field_id
    if field_id in {"received_at", "date", "sample_received"} and draft.source in {"date", "trocr"}:
        canonical_id = field_id
    hyps = [{"value": draft.text, "score": round(draft.confidence, 3), "source": draft.source}]
    for alt, score in draft.alternatives:
        if alt and alt != draft.text:
            hyps.append({"value": alt, "score": round(float(score), 3), "source": "alt"})
    return HandwritingPrediction(
        field_id=field_id,
        raw_text=draft.text,
        canonical_value=canonical,
        canonical_id=canonical_id,
        confidence=round(float(draft.confidence), 3),
        source=draft.source,
        needs_hitl=needs_hitl,
        hypotheses=hyps,
    )


def recognize_fields(
    crops: dict[str, np.ndarray | None],
    *,
    backend: str = "trocr",
    blanks: dict[str, np.ndarray | None] | None = None,
) -> dict[str, HandwritingPrediction]:
    """Recognize many handwriting crops. Independent of PaddleOCR / nonverbal."""
    blanks = blanks or {}
    return {
        fid: recognize_verbal(crop, fid, backend=backend, blank=blanks.get(fid))
        for fid, crop in crops.items()
    }


def attach_kg_priors(
    prediction: HandwritingPrediction,
    kg: KnowledgeGraph | None,
    *,
    ticked_ids: list[str] | None = None,
    expected_tubes: dict[str, int] | None = None,
) -> HandwritingPrediction:
    """Optional Block 2 `assume()` / tube prior — Block 4, not the live Block 3 path."""
    if kg is None:
        return prediction
    return fuse_handwriting(
        prediction.field_id,
        prediction.raw_text,
        prediction.confidence,
        prediction.source,
        kg=kg,
        ticked_ids=ticked_ids or [],
        expected_tubes=expected_tubes or {},
    )


def attach_kg_priors_batch(
    fields: dict[str, HandwritingPrediction],
    kg: KnowledgeGraph | None,
    *,
    ticked_ids: list[str] | None = None,
    expected_tubes: dict[str, int] | None = None,
) -> dict[str, HandwritingPrediction]:
    if kg is None:
        return fields
    return {
        fid: attach_kg_priors(pred, kg, ticked_ids=ticked_ids, expected_tubes=expected_tubes)
        for fid, pred in fields.items()
    }

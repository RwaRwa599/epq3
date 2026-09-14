"""Diagnostic overall_confidence from 1c retry/HITL and empty tube crops."""

from __future__ import annotations

from typing import Any

from med_doc.htr.schemas import DocumentHypotheses, DocumentPrediction, HandwritingPrediction


def _empty_hw(field: HandwritingPrediction) -> bool:
    return not (field.raw_text or "").strip() or field.source == "empty"


def crop_quality(
    hyp: DocumentHypotheses | None = None,
    pred: DocumentPrediction | None = None,
) -> dict[str, Any]:
    """Rates that actually move on clinic dumps (1c retry/HITL, empty tubes)."""
    cv: dict[str, Any] = {}
    nonverbal = {}
    verbal = {}
    ticked: list[str] = []
    if hyp is not None:
        cv = dict(hyp.crop_validate or {})
        nonverbal = hyp.nonverbal
        verbal = hyp.verbal
        ticked = list(hyp.ticked_test_ids)
    if pred is not None:
        cv = dict(pred.crop_validate or cv)
        nonverbal = pred.checkbox_marks or nonverbal
        verbal = pred.handwriting_fields or verbal
        ticked = list(pred.ticked_test_ids or ticked)

    n_ok = int(cv.get("n_ok") or 0)
    n_retry = int(cv.get("n_retry") or 0)
    n_hitl = int(cv.get("n_hitl") or 0)
    n_skip = int(cv.get("n_skip") or 0)
    n_cv = n_ok + n_retry + n_hitl + n_skip
    if n_cv == 0 and nonverbal:
        statuses = [(m.crop_validate_status or "") for m in nonverbal.values()]
        n_retry = sum(1 for s in statuses if s == "retry")
        n_hitl = sum(1 for s in statuses if s == "hitl")
        n_ok = sum(1 for s in statuses if s == "ok")
        n_skip = sum(1 for s in statuses if s == "skip")
        n_cv = len(statuses)
        if n_retry + n_hitl + n_ok + n_skip == 0:
            n_hitl = sum(1 for m in nonverbal.values() if m.needs_hitl)
            n_cv = len(nonverbal)

    retry_rate = (n_retry / n_cv) if n_cv else 0.0
    hitl_rate = (n_hitl / n_cv) if n_cv else 0.0
    tubes = [f for fid, f in verbal.items() if fid.startswith("tube_")]
    empty_crop_rate = (
        (sum(1 for f in tubes if _empty_hw(f)) / len(tubes)) if tubes else 0.0
    )
    all_tube_crops_empty = bool(tubes) and all(_empty_hw(f) for f in tubes)

    conf = 1.0 - 0.35 * retry_rate - 0.55 * hitl_rate
    conf *= 1.0 - 0.45 * empty_crop_rate
    n_cb = n_cv if n_cv else len(nonverbal)
    if n_cb >= 20:
        conf *= 1.0 - 0.4 * min(1.0, len(ticked) / n_cb)
    if ticked and all_tube_crops_empty:
        conf = min(conf, 0.2)
    conf = max(0.0, min(1.0, conf))

    return {
        "n_ok": n_ok,
        "n_retry": n_retry,
        "n_hitl": n_hitl,
        "n_skip": n_skip,
        "n_checkbox": n_cv or n_cb,
        "retry_rate": round(retry_rate, 4),
        "hitl_rate": round(hitl_rate, 4),
        "empty_crop_rate": round(empty_crop_rate, 4),
        "all_tube_crops_empty": all_tube_crops_empty,
        "overall_confidence": round(conf, 3),
    }


def diagnostic_confidence(hyp: DocumentHypotheses) -> float:
    return float(crop_quality(hyp=hyp)["overall_confidence"])

"""Page-level registration gates shared by Blocks 1a and 1c."""

from __future__ import annotations

from typing import Any

# Clinic photos at 0.36–0.54 still emitted "success". Below this, do not trust the page.
MIN_ALIGNMENT_CONFIDENCE = 0.6
TEMPLATE_SCORE_MARGIN = 2.0
TEMPLATE_RELATIVE_MARGIN = 0.10
# Below this peak, a v0/v1 tie is "both templates failed", not a close call.
TEMPLATE_MIN_PEAK_SCORE = 1.0


def alignment_gate(confidence: float, *, threshold: float = MIN_ALIGNMENT_CONFIDENCE) -> dict[str, Any]:
    conf = float(confidence or 0.0)
    ok = conf >= threshold
    return {
        "ok": ok,
        "reason": None if ok else "alignment_confidence",
        "alignment_confidence": round(conf, 4),
        "threshold": threshold,
    }


def template_pick_meta(scores: dict[str, Any]) -> dict[str, Any]:
    """Flag visually similar forms that resolve to different print revisions."""
    s0 = float(scores.get("score_v0") or 0.0)
    s1 = float(scores.get("score_v1") or 0.0)
    picked = str(scores.get("picked") or "")
    margin = abs(s1 - s0)
    peak = max(s0, s1)
    relative = margin / peak if peak > 0 else 0.0
    both_failed = peak < TEMPLATE_MIN_PEAK_SCORE
    # Tied at ~0 is total registration failure, not template ambiguity.
    ambiguous = (not both_failed) and (
        margin < TEMPLATE_SCORE_MARGIN or relative < TEMPLATE_RELATIVE_MARGIN
    )
    if both_failed:
        extra_note = "; both templates failed (scores ~0) — registration, not a close pick"
    elif ambiguous:
        extra_note = "; scores close — verify template_id"
    else:
        extra_note = ""
    return {
        **scores,
        "score_margin": round(margin, 3),
        "relative_margin": round(relative, 4),
        "ambiguous": ambiguous,
        "both_failed": both_failed,
        "note": (
            f"picked {picked} (v0={s0:.1f} v1={s1:.1f}, margin={margin:.1f}, "
            f"rel={relative:.1%})"
            + extra_note
        ),
    }


def template_pick_needs_review(pick: dict[str, Any] | None) -> bool:
    pick = pick or {}
    return bool(pick.get("ambiguous") or pick.get("both_failed"))


def batch_template_conflict(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Same batch, different template_id → likely a picker miss, not two form versions."""
    ids = [str(d.get("template_id") or "") for d in documents if d.get("status") != "error"]
    unique = sorted({i for i in ids if i})
    return {
        "conflict": len(unique) > 1,
        "template_ids": unique,
        "n_documents": len(ids),
    }

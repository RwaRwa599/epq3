"""Page-level registration gates shared by Blocks 1a and 1c."""

from __future__ import annotations

from typing import Any

# Clinic photos at 0.36–0.54 still emitted "success". Below this, do not trust the page.
MIN_ALIGNMENT_CONFIDENCE = 0.6
TEMPLATE_SCORE_MARGIN = 2.0
TEMPLATE_RELATIVE_MARGIN = 0.10


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
    peak = max(s0, s1, 1e-6)
    relative = margin / peak
    ambiguous = margin < TEMPLATE_SCORE_MARGIN or relative < TEMPLATE_RELATIVE_MARGIN
    return {
        **scores,
        "score_margin": round(margin, 3),
        "relative_margin": round(relative, 4),
        "ambiguous": ambiguous,
        "note": (
            f"picked {picked} (v0={s0:.1f} v1={s1:.1f}, margin={margin:.1f}, "
            f"rel={relative:.1%})"
            + ("; scores close — verify template_id" if ambiguous else "")
        ),
    }


def batch_template_conflict(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Same batch, different template_id → likely a picker miss, not two form versions."""
    ids = [str(d.get("template_id") or "") for d in documents if d.get("status") != "error"]
    unique = sorted({i for i in ids if i})
    return {
        "conflict": len(unique) > 1,
        "template_ids": unique,
        "n_documents": len(ids),
    }

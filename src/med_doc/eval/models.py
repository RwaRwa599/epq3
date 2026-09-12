"""What actually runs for ticks vs handwriting. Not a VLM stack."""

from __future__ import annotations

from typing import Any

from med_doc.htr.marks import TICK_POLICY
from med_doc.htr.nonverbal import paddle_available
from med_doc.htr.verbal import trocr_available


def models_in_use(*, verbal_backend: str = "lexicon", mark_backend: str = "geometry") -> dict[str, Any]:
    """Live stack for a pipeline run. Tick CNN / TrOCR fine-tunes are not in this path."""
    ticks = "geometry+logreg (residual vs blank template patch)"
    if mark_backend == "paddle":
        ticks = "geometry+logreg fused with PaddleOCR whitelist (opt-in; over-calls printed glyphs)"
    verbal = "digits/charset/write-in lexicon (glyphs.py)"
    if verbal_backend in ("trocr", "auto"):
        verbal = "glyphs first; TrOCR only if installed and glyphs empty"
    if verbal_backend == "lexicon":
        verbal = "glyphs + visual lexicon only (TrOCR not loaded)"
    return {
        "tick_policy": TICK_POLICY,
        "ticks": ticks,
        "mark_backend": mark_backend,
        "paddle_installed": paddle_available(),
        "paddle_used_for_ticks": mark_backend == "paddle" and paddle_available(),
        "verbal": verbal,
        "verbal_backend": verbal_backend,
        "trocr_installed": trocr_available(),
        "trocr_used": verbal_backend in ("trocr", "auto") and trocr_available(),
        "crop_cnn": "not wired — use only if logreg plateaus on labeled clinic empties",
        "llm": "Block 5 n-best rank stub only; never classifies ticks",
        "right_model": (
            "Checkboxes: logistic on mark_features, not TrOCR/Paddle/LLM. "
            "Tubes/dates: glyph matcher. Fine-tune TrOCR only after ticks stabilize."
        ),
    }

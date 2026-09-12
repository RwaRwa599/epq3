"""Nonverbal Block 3: checkbox mark classification (PaddleOCR, density fallback).

This path never imports TrOCR. When PaddleOCR is missing or fails on a crop,
classification falls back to the density / slash / filled detector.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import cv2
import numpy as np

from med_doc.htr.marks import SLASH_MIN_DENSITY, classify_mark, looks_like_text_line
from med_doc.htr.schemas import MarkPrediction

logger = logging.getLogger(__name__)

_PADDLE: Any = None
_PADDLE_FAILED = False

MARK_TOKENS = {
    "v",
    "x",
    "/",
    "\\",
    "+",
    "*",
    "o",
    "ok",
    "tick",
    "check",
    "yes",
    "✓",
    "✔",
    "√",
    "×",
    "✕",
    "✗",
    "✘",
}


def paddle_available() -> bool:
    """True when PaddleOCR can be constructed in this process."""
    return _get_paddle() is not None


def _get_paddle() -> Any:
    global _PADDLE, _PADDLE_FAILED
    if _PADDLE_FAILED:
        return None
    if _PADDLE is not None:
        return _PADDLE
    try:
        os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        logging.getLogger("paddle").setLevel(logging.ERROR)
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as exc:
        logger.info("PaddleOCR unavailable (%s); using density fallback.", exc)
        _PADDLE_FAILED = True
        return None

    try:
        try:
            _PADDLE = PaddleOCR(
                lang="en",
                use_angle_cls=False,
                show_log=False,
                use_gpu=False,
            )
        except TypeError:
            _PADDLE = PaddleOCR(lang="en")
    except Exception as exc:
        logger.info("PaddleOCR init failed (%s); using density fallback.", exc)
        _PADDLE_FAILED = True
        return None
    return _PADDLE


def _upscale_crop(crop: np.ndarray, min_side: int = 128) -> np.ndarray:
    arr = np.asarray(crop)
    if arr.size == 0:
        return arr
    h, w = arr.shape[:2]
    scale = max(1.0, float(min_side) / max(1, min(h, w)))
    if scale > 1.01:
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    pad = 8
    if arr.ndim == 2:
        return cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def _paddle_hits(engine: Any, crop: np.ndarray) -> list[tuple[str, float]]:
    """Return (text, score) pairs from PaddleOCR 2.x `ocr` or 3.x `predict`."""
    img = _upscale_crop(crop)
    if img.size == 0:
        return []
    hits: list[tuple[str, float]] = []

    if hasattr(engine, "ocr"):
        try:
            raw = engine.ocr(img, cls=False)
        except TypeError:
            raw = engine.ocr(img)
        except Exception:
            raw = None
        if raw:
            pages = raw if isinstance(raw, list) else [raw]
            for page in pages:
                if not page:
                    continue
                for item in page:
                    if item is None:
                        continue
                    text, score = "", 0.0
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        rec = item[1]
                        if isinstance(rec, (list, tuple)) and rec:
                            text = str(rec[0] or "")
                            try:
                                score = float(rec[1])
                            except (TypeError, ValueError):
                                score = 0.5
                        elif isinstance(rec, str):
                            text = rec
                            score = 0.5
                    hits.append((text.strip(), score))
            if hits:
                return hits

    if hasattr(engine, "predict"):
        try:
            pred = engine.predict(img)
        except Exception:
            pred = None
        rows = pred if isinstance(pred, list) else [pred]
        for row in rows:
            if not isinstance(row, dict):
                continue
            texts = row.get("rec_texts") or row.get("rec_text") or []
            scores = row.get("rec_scores") or row.get("rec_score") or []
            if isinstance(texts, str):
                texts = [texts]
            if not isinstance(scores, (list, tuple)):
                scores = [scores]
            for i, text in enumerate(texts):
                score = float(scores[i]) if i < len(scores) else 0.5
                hits.append((str(text).strip(), score))
    return hits


def _looks_like_mark(text: str) -> bool:
    raw = (text or "").strip().lower()
    if not raw:
        return False
    if raw in MARK_TOKENS:
        return True
    if len(raw) <= 3 and any(ch in MARK_TOKENS for ch in raw):
        return True
    return False


def classify_mark_nonverbal(
    crop: np.ndarray | None,
    field_id: str,
    *,
    fallback_dark_ratio: float | None = None,
    fallback_candidate: bool | None = None,
) -> MarkPrediction:
    """Classify one checkbox. Source is `paddle` or `density_fallback`."""
    density_pred = classify_mark(
        crop,
        field_id,
        fallback_dark_ratio=fallback_dark_ratio,
        fallback_candidate=fallback_candidate,
    )
    density_labeled = density_pred.model_copy(update={"source": "density_fallback"})

    if crop is None or (hasattr(crop, "size") and np.asarray(crop).size == 0):
        return density_labeled

    engine = _get_paddle()
    if engine is None:
        return density_labeled

    try:
        hits = _paddle_hits(engine, np.asarray(crop))
    except Exception as exc:
        logger.info("PaddleOCR failed on %s (%s); density fallback.", field_id, exc)
        return density_labeled

    whitelist = [(t, s) for t, s in hits if _looks_like_mark(t)]
    dens = float(density_pred.ink_density)
    in_band = SLASH_MIN_DENSITY <= dens < 0.62
    text_line = looks_like_text_line(crop)

    # Geometry alone is enough (Paddle off must still recall V/slash/fill).
    if density_pred.is_marked and density_pred.source in {"slash", "v_check", "filled"}:
        if whitelist:
            best = max(whitelist, key=lambda x: x[1])
            conf = float(min(0.97, max(density_pred.confidence, best[1])))
            return density_pred.model_copy(update={"source": "paddle", "confidence": round(conf, 3)})
        return density_labeled

    # Second vote: whitelist token + interior ink + not a label strip (no score≥0.55 loophole).
    if whitelist and in_band and not text_line:
        best = max(whitelist, key=lambda x: x[1])
        conf = float(min(0.97, max(0.6, best[1])))
        return MarkPrediction(
            field_id=field_id,
            is_marked=True,
            confidence=round(conf, 3),
            ink_density=density_pred.ink_density,
            needs_hitl=conf < 0.70,
            source="paddle",
        )

    return MarkPrediction(
        field_id=field_id,
        is_marked=False,
        confidence=round(max(density_pred.confidence, 0.85), 3),
        ink_density=density_pred.ink_density,
        needs_hitl=False if not density_pred.is_marked else density_pred.needs_hitl,
        source="paddle" if engine is not None else "density_fallback",
    )


def classify_marks(
    crops: dict[str, np.ndarray | None],
    *,
    fallbacks: dict[str, dict[str, Any]] | None = None,
) -> dict[str, MarkPrediction]:
    """Classify many checkbox crops. Independent of TrOCR / verbal."""
    fallbacks = fallbacks or {}
    out: dict[str, MarkPrediction] = {}
    for fid, crop in crops.items():
        info = fallbacks.get(fid) or {}
        out[fid] = classify_mark_nonverbal(
            crop,
            fid,
            fallback_dark_ratio=info.get("dark_ratio"),
            fallback_candidate=info.get("is_marked_candidate"),
        )
    return out

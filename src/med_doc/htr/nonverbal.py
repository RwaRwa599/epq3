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

from med_doc.htr.marks import classify_mark, looks_like_text_line, logreg_mark_prob, mark_features
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


def _geometry_prob(pred) -> float:
    if pred.source in {"slash", "v_check", "filled"}:
        return float(max(0.78, pred.confidence))
    if pred.is_marked:
        return float(np.clip(pred.confidence, 0.55, 0.95))
    if pred.source == "logreg":
        return float(np.clip(1.0 - pred.confidence, 0.05, 0.45))
    return 0.12


def _fuse_probs(p_geom: float, p_paddle: float) -> float:
    """Weighted average plus noisy-OR so two weak votes can reinforce."""
    mix = 0.68 * p_geom + 0.32 * p_paddle
    noisy_or = 1.0 - (1.0 - p_geom) * (1.0 - p_paddle)
    return float(np.clip(0.55 * mix + 0.45 * noisy_or, 0.0, 1.0))


def classify_mark_nonverbal(
    crop: np.ndarray | None,
    field_id: str,
    *,
    fallback_dark_ratio: float | None = None,
    fallback_candidate: bool | None = None,
    blank: np.ndarray | None = None,
) -> MarkPrediction:
    """Classify one checkbox. Geometry and Paddle are fused when both exist."""
    density_pred = classify_mark(
        crop,
        field_id,
        fallback_dark_ratio=fallback_dark_ratio,
        fallback_candidate=fallback_candidate,
        blank=blank,
    )
    density_labeled = density_pred.model_copy(update={"source": "density_fallback"})

    if crop is None or (hasattr(crop, "size") and np.asarray(crop).size == 0):
        return density_labeled

    p_geom = _geometry_prob(density_pred)
    if density_pred.source == "logreg" or density_pred.source in {"slash", "v_check", "filled"}:
        try:
            p_geom = max(p_geom, logreg_mark_prob(mark_features(np.asarray(crop), blank=blank)))
        except Exception:
            pass

    engine = _get_paddle()
    if engine is None:
        return density_pred.model_copy(update={"source": "density_fallback"})

    try:
        hits = _paddle_hits(engine, np.asarray(crop))
    except Exception as exc:
        logger.info("PaddleOCR failed on %s (%s); density fallback.", field_id, exc)
        return density_labeled

    whitelist = [(t, s) for t, s in hits if _looks_like_mark(t)]
    p_paddle = float(max((s for _t, s in whitelist), default=0.0))
    text_line = looks_like_text_line(crop)
    if text_line:
        p_paddle *= 0.25

    p = _fuse_probs(p_geom, p_paddle)
    # Precision: a weak Paddle token must not override a clearly empty geometry vote.
    if p_geom < 0.22 and p_paddle < 0.85:
        p = min(p, p_geom)

    marked = p >= 0.50
    if density_pred.source in {"slash", "v_check", "filled"} and p_geom >= 0.70:
        marked = True
        p = max(p, p_geom)

    source = "paddle" if p_paddle >= 0.35 and marked else (
        density_pred.source if density_pred.source not in {"density", "density_fallback"} else "fused"
    )
    if not marked and engine is not None:
        source = "fused" if p_paddle > 0 else density_pred.source

    return MarkPrediction(
        field_id=field_id,
        is_marked=bool(marked),
        confidence=round(float(np.clip(p if marked else max(0.55, 1.0 - p), 0.0, 1.0)), 3),
        ink_density=density_pred.ink_density,
        needs_hitl=0.42 <= p <= 0.58 or density_pred.needs_hitl,
        source=source,
    )


def classify_marks(
    crops: dict[str, np.ndarray | None],
    *,
    fallbacks: dict[str, dict[str, Any]] | None = None,
    template: Any | None = None,
    canvas_size: tuple[int, int] | list[int] | None = None,
) -> dict[str, MarkPrediction]:
    """Classify many checkbox crops. Independent of TrOCR / verbal."""
    from med_doc.htr.blank import blank_patch

    fallbacks = fallbacks or {}
    use_blank = (
        template is not None
        and canvas_size is not None
        and int(canvas_size[0]) == int(template.width)
        and int(canvas_size[1]) == int(template.height)
    )
    out: dict[str, MarkPrediction] = {}
    for fid, crop in crops.items():
        info = fallbacks.get(fid) or {}
        blank = None
        if use_blank and crop is not None and getattr(crop, "size", 0):
            arr = np.asarray(crop)
            if arr.size:
                bbox = info.get("bbox") or info.get("canonical_bbox")
                blank = blank_patch(
                    template,
                    fid,
                    (arr.shape[0], arr.shape[1]),
                    bbox=bbox if bbox else None,
                )
        out[fid] = classify_mark_nonverbal(
            crop,
            fid,
            fallback_dark_ratio=info.get("dark_ratio"),
            fallback_candidate=info.get("is_marked_candidate"),
            blank=blank,
        )
    return out

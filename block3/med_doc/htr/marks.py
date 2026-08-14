"""Checkbox mark classifier: ink density, slash detection, shadow rejection."""

from __future__ import annotations

import numpy as np

from med_doc.htr.schemas import MarkPrediction

DENSITY_TAU = 0.08
BORDERLINE_HI = 0.15
HITL_CONFIDENCE = 0.70


def _as_gray(crop: np.ndarray) -> np.ndarray:
    arr = np.asarray(crop)
    if arr.ndim == 3:
        return arr.mean(axis=2).astype(np.float32)
    return arr.astype(np.float32)


def ink_density(crop: np.ndarray, threshold: float = 140.0) -> float:
    """Fraction of dark pixels in the checkbox interior, ignoring a thin border."""
    gray = _as_gray(crop)
    if gray.size == 0:
        return 0.0
    h, w = gray.shape[:2]
    y0, y1 = int(h * 0.22), int(h * 0.78)
    x0, x1 = int(w * 0.22), int(w * 0.78)
    interior = gray[y0:y1, x0:x1] if y1 > y0 and x1 > x0 else gray
    if interior.size == 0:
        return 0.0
    return float((interior < threshold).mean())


def _hollow_empty(gray: np.ndarray) -> bool:
    """True when the crop looks like an empty printed square (dark ring, bright center)."""
    if gray.size == 0 or min(gray.shape[:2]) < 6:
        return False
    border = np.concatenate([gray[0], gray[-1], gray[1:-1, 0], gray[1:-1, -1]])
    interior = gray[2:-2, 2:-2]
    if interior.size == 0:
        return False
    return float(interior.mean()) - float(border.mean()) > 8.0 and float(interior.mean()) > 160.0


def _diagonal_stroke(gray: np.ndarray) -> bool:
    """True when dark pixels form a slash rather than printed-letter blobs."""
    if gray.size == 0 or min(gray.shape[:2]) < 6:
        return False
    interior = gray[1:-1, 1:-1]
    ink = interior < 140
    if float(ink.mean()) < 0.05:
        return False
    ys, xs = np.where(ink)
    if len(xs) < 4 or float(xs.std()) < 1e-6 or float(ys.std()) < 1e-6:
        return False
    corr = abs(float(np.corrcoef(xs.astype(float), ys.astype(float))[0, 1]))
    return corr >= 0.55


def classify_mark(
    crop: np.ndarray | None,
    field_id: str,
    *,
    density_tau: float = DENSITY_TAU,
    fallback_dark_ratio: float | None = None,
    fallback_candidate: bool | None = None,
) -> MarkPrediction:
    """Classify a checkbox crop as marked or empty.

    Falls back to Block 1 metadata heuristics when the crop image is missing.
    Borderline densities (tau .. borderline_hi) are flagged for HiTL review.
    """
    if crop is None or (hasattr(crop, "size") and np.asarray(crop).size == 0):
        density = float(fallback_dark_ratio or 0.0)
        marked = bool(fallback_candidate) if fallback_candidate is not None else density >= density_tau
        near = abs(density - density_tau) < 0.04
        conf = float(min(0.95, 0.55 + abs(density - density_tau) * 3.0))
        return MarkPrediction(
            field_id=field_id,
            is_marked=marked,
            confidence=round(conf, 3),
            ink_density=round(density, 4),
            needs_hitl=near,
            source="metadata_fallback",
        )

    gray = _as_gray(crop)
    density = ink_density(crop)
    mean = float(gray.mean()) if gray.size else 255.0

    # Very dark crop is likely a shadow/desk bleed, not a tick.
    if mean < 90.0:
        return MarkPrediction(
            field_id=field_id,
            is_marked=False,
            confidence=0.85,
            ink_density=round(density, 4),
            needs_hitl=False,
            source="shadow-crop",
        )

    slash = _diagonal_stroke(gray)
    filled = density >= 0.45
    density_marked = density >= density_tau

    if slash or filled:
        source = "slash" if slash and not filled else "filled"
        conf = 0.9 if slash else 0.92
        return MarkPrediction(
            field_id=field_id,
            is_marked=True,
            confidence=conf,
            ink_density=round(density, 4),
            needs_hitl=False,
            source=source,
        )

    if _hollow_empty(gray) and density < density_tau:
        return MarkPrediction(
            field_id=field_id,
            is_marked=False,
            confidence=0.9,
            ink_density=round(density, 4),
            needs_hitl=False,
            source="hollow-empty",
        )

    marked = bool(density_marked)
    dist = abs(density - density_tau)
    conf = float(min(0.95, 0.55 + dist * 3.0))
    near_threshold = density_tau <= density <= BORDERLINE_HI
    needs_hitl = near_threshold or conf < HITL_CONFIDENCE

    return MarkPrediction(
        field_id=field_id,
        is_marked=marked,
        confidence=round(conf, 3),
        ink_density=round(density, 4),
        needs_hitl=bool(needs_hitl),
        source="density",
    )

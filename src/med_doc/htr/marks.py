"""Checkbox mark classifier: interior ink only (printed-frame slash is not a tick)."""

from __future__ import annotations

import cv2
import numpy as np

from med_doc.htr.schemas import MarkPrediction

# Interior-only density. Clinic empty boxes sit well below this; a real slash is above.
DENSITY_TAU = 0.10
SLASH_MIN_DENSITY = 0.06
BORDERLINE_HI = 0.18
HITL_CONFIDENCE = 0.70
INSET = 0.28


def _as_gray(crop: np.ndarray) -> np.ndarray:
    arr = np.asarray(crop)
    if arr.ndim == 3:
        return arr.mean(axis=2).astype(np.float32)
    return arr.astype(np.float32)


def _interior(gray: np.ndarray, inset: float = INSET) -> np.ndarray:
    if gray.size == 0:
        return gray
    h, w = gray.shape[:2]
    y0, y1 = int(h * inset), int(h * (1.0 - inset))
    x0, x1 = int(w * inset), int(w * (1.0 - inset))
    if y1 <= y0 or x1 <= x0:
        return gray
    return gray[y0:y1, x0:x1]


def _ink_cut(interior: np.ndarray, threshold: float = 140.0) -> float:
    """Paper-relative cut so gray photos don't count as ink."""
    if interior.size == 0:
        return threshold
    paper = float(np.percentile(interior, 88))
    return float(min(threshold, max(40.0, paper - 40.0)))


def ink_density(crop: np.ndarray, threshold: float = 140.0) -> float:
    """Fraction of dark pixels in the checkbox *interior*, ignoring the printed frame."""
    gray = _as_gray(crop)
    interior = _interior(gray)
    if interior.size == 0:
        return 0.0
    cut = _ink_cut(interior, threshold)
    return float((interior < cut).mean())


def _hollow_empty(gray: np.ndarray) -> bool:
    """True when the crop is a printed square: dark ring, bright interior."""
    if gray.size == 0 or min(gray.shape[:2]) < 8:
        return False
    interior = _interior(gray)
    if interior.size == 0:
        return False
    border = np.concatenate([gray[0], gray[-1], gray[1:-1, 0], gray[1:-1, -1]])
    return float(interior.mean()) - float(border.mean()) > 12.0 and float(interior.mean()) > 150.0


def _border_dark_frac(gray: np.ndarray) -> float:
    if gray.size == 0 or min(gray.shape[:2]) < 6:
        return 0.0
    band = 3
    border = np.concatenate(
        [gray[:band].ravel(), gray[-band:].ravel(), gray[:, :band].ravel(), gray[:, -band:].ravel()]
    )
    cut = _ink_cut(gray)
    return float((border < cut).mean())


def _ink_blob_count(gray: np.ndarray) -> int:
    """How many interior ink blobs (specks ignored). A handwritten tick is one stroke."""
    interior = _interior(gray)
    if interior.size == 0:
        return 0
    cut = _ink_cut(interior)
    ink = (interior < cut).astype(np.uint8)
    n, labels = cv2.connectedComponents(ink)
    return sum(1 for i in range(1, n) if int((labels == i).sum()) >= 5)


def _diagonal_stroke(gray: np.ndarray) -> bool:
    """True when *interior* dark pixels form a single slash. Printed corners/letters excluded."""
    interior = _interior(gray)
    if interior.size == 0 or min(interior.shape[:2]) < 6:
        return False
    if _ink_blob_count(gray) != 1:
        return False
    cut = _ink_cut(interior)
    ink = interior < cut
    if float(ink.mean()) < SLASH_MIN_DENSITY:
        return False
    ys, xs = np.where(ink)
    if len(xs) < 6 or float(xs.std()) < 1e-6 or float(ys.std()) < 1e-6:
        return False
    corr = abs(float(np.corrcoef(xs.astype(float), ys.astype(float))[0, 1]))
    return corr >= 0.70


def classify_mark(
    crop: np.ndarray | None,
    field_id: str,
    *,
    density_tau: float = DENSITY_TAU,
    fallback_dark_ratio: float | None = None,
    fallback_candidate: bool | None = None,
) -> MarkPrediction:
    """Classify a checkbox crop as marked or empty.

    Precision-first: a printed frame or L-corner is unmarked. A mark needs interior
    ink (filled box, slash through the interior, or density ≥ tau).
    Falls back to Block 1 metadata when the crop image is missing.
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

    if mean < 90.0:
        return MarkPrediction(
            field_id=field_id,
            is_marked=False,
            confidence=0.85,
            ink_density=round(density, 4),
            needs_hitl=False,
            source="shadow-crop",
        )

    # No interior ink → empty, even if the printed frame looks diagonal.
    if density < SLASH_MIN_DENSITY:
        source = "hollow-empty" if _hollow_empty(gray) else "density"
        return MarkPrediction(
            field_id=field_id,
            is_marked=False,
            confidence=0.95,
            ink_density=round(density, 4),
            needs_hitl=False,
            source=source,
        )

    slash = _diagonal_stroke(gray)
    filled = density >= 0.62
    border = _border_dark_frac(gray)
    # A real tick sits in a printed square: modest frame, not a letter crop.
    checkbox_frame = 0.055 <= border <= 0.12
    small_printed_box = _hollow_empty(gray) and border >= 0.25

    if (slash and (checkbox_frame or small_printed_box)) or filled:
        source = "slash" if slash and not filled else "filled"
        conf = 0.93 if slash else 0.94
        return MarkPrediction(
            field_id=field_id,
            is_marked=True,
            confidence=conf,
            ink_density=round(density, 4),
            needs_hitl=False,
            source=source,
        )

    if _hollow_empty(gray):
        return MarkPrediction(
            field_id=field_id,
            is_marked=False,
            confidence=0.95,
            ink_density=round(density, 4),
            needs_hitl=False,
            source="hollow-empty",
        )

    # Interior ink that is not a slash or a filled box is printed label / mis-crop, not a tick.
    return MarkPrediction(
        field_id=field_id,
        is_marked=False,
        confidence=0.9,
        ink_density=round(density, 4),
        needs_hitl=False,
        source="density",
    )

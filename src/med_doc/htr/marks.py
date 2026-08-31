"""Checkbox mark classifier: interior ink only (printed-frame slash is not a tick)."""

from __future__ import annotations

import cv2
import numpy as np

from med_doc.htr.schemas import MarkPrediction

# Interior-only density. Clinic empty boxes sit well below this; a real slash is above.
DENSITY_TAU = 0.10
SLASH_MIN_DENSITY = 0.06
FILL_DENSITY = 0.62
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


def _annulus_dark_frac(gray: np.ndarray) -> float:
    """Dark fraction of a ring around the inset interior — not the PNG edge.

    1b pads the square with paper, so the printed frame sits inside the crop.
    """
    if gray.size == 0 or min(gray.shape[:2]) < 10:
        return 0.0
    h, w = gray.shape[:2]
    outer = 0.12
    inner = INSET
    y0, y1 = int(h * outer), int(h * (1.0 - outer))
    x0, x1 = int(w * outer), int(w * (1.0 - outer))
    iy0, iy1 = int(h * inner), int(h * (1.0 - inner))
    ix0, ix1 = int(w * inner), int(w * (1.0 - inner))
    if y1 <= y0 or x1 <= x0 or iy1 <= iy0 or ix1 <= ix0:
        return 0.0
    band = gray[y0:y1, x0:x1].copy()
    band[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = 255
    cut = _ink_cut(gray)
    return float((band < cut).mean())


def _border_dark_frac(gray: np.ndarray) -> float:
    """PNG-edge dark fraction (unit-test crops). Prefer `_annulus_dark_frac` in product."""
    if gray.size == 0 or min(gray.shape[:2]) < 6:
        return 0.0
    band = 3
    border = np.concatenate(
        [gray[:band].ravel(), gray[-band:].ravel(), gray[:, :band].ravel(), gray[:, -band:].ravel()]
    )
    cut = _ink_cut(gray)
    return float((border < cut).mean())


def _ink_mask(interior: np.ndarray) -> np.ndarray:
    if interior.size == 0:
        return np.zeros((0, 0), dtype=bool)
    cut = _ink_cut(interior)
    return interior < cut


def _ink_blob_count(gray: np.ndarray) -> int:
    """How many interior ink blobs (specks ignored). A handwritten tick is one or two strokes."""
    interior = _interior(gray)
    if interior.size == 0:
        return 0
    ink = _ink_mask(interior).astype(np.uint8)
    n, labels = cv2.connectedComponents(ink)
    return sum(1 for i in range(1, n) if int((labels == i).sum()) >= 5)


def looks_like_text_line(crop: np.ndarray) -> bool:
    """True when ink is a horizontal label strip, not a mark inside a square."""
    gray = _as_gray(crop)
    if gray.size == 0 or min(gray.shape[:2]) < 8:
        return False
    h, w = gray.shape[:2]
    cut = _ink_cut(gray)
    dark = gray < cut
    dens = float(dark.mean())
    if dens < 0.03:
        return False
    if w >= int(h * 1.45) and dens >= 0.03:
        return True
    row = dark.mean(axis=1)
    wide = row >= 0.32
    if int(wide.sum()) < 2:
        return False
    col = dark.mean(axis=0)
    return float(col.std()) < 0.14 and float(wide.mean()) >= 0.10


def _diagonal_stroke(gray: np.ndarray) -> bool:
    """True when *interior* dark pixels form a single slash."""
    interior = _interior(gray)
    if interior.size == 0 or min(interior.shape[:2]) < 6:
        return False
    if _ink_blob_count(gray) != 1:
        return False
    ink = _ink_mask(interior)
    if float(ink.mean()) < SLASH_MIN_DENSITY:
        return False
    ys, xs = np.where(ink)
    if len(xs) < 6 or float(xs.std()) < 1e-6 or float(ys.std()) < 1e-6:
        return False
    if float(xs.std()) > 2.8 * float(ys.std()):
        return False
    corr = abs(float(np.corrcoef(xs.astype(float), ys.astype(float))[0, 1]))
    return corr >= 0.70


def _v_or_check_stroke(gray: np.ndarray) -> bool:
    """V / check / lambda: two arms in the interior, not a single diagonal and not a glyph line."""
    interior = _interior(gray)
    if interior.size == 0 or min(interior.shape[:2]) < 6:
        return False
    blobs = _ink_blob_count(gray)
    if blobs < 1 or blobs > 2:
        return False
    ink = _ink_mask(interior)
    dens = float(ink.mean())
    if dens < SLASH_MIN_DENSITY or dens >= FILL_DENSITY:
        return False
    ys, xs = np.where(ink)
    if len(xs) < 8:
        return False
    ih, iw = interior.shape[:2]
    span_x = (int(xs.max()) - int(xs.min()) + 1) / float(max(iw, 1))
    span_y = (int(ys.max()) - int(ys.min()) + 1) / float(max(ih, 1))
    if span_x < 0.28 or span_y < 0.28:
        return False
    if float(xs.std()) > 2.8 * max(float(ys.std()), 1e-6):
        return False
    cx = float(np.median(xs))
    left = ys[xs < cx]
    right = ys[xs >= cx]
    if len(left) < 3 or len(right) < 3:
        return False
    left_xs = xs[xs < cx].astype(float)
    right_xs = xs[xs >= cx].astype(float)
    left_ys = left.astype(float)
    right_ys = right.astype(float)

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if float(a.std()) < 1e-6 or float(b.std()) < 1e-6:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    cl = _corr(left_xs, left_ys)
    cr = _corr(right_xs, right_ys)
    # V / check: arms lean opposite, or one slash plus a short opposing stroke.
    opposite = (cl * cr) < -0.05 or (cl > 0.25 and cr < -0.15) or (cl < -0.15 and cr > 0.25)
    check_long = abs(cr) >= 0.45 and len(right) >= len(left)
    lambda_like = abs(cl) >= 0.40 and abs(cr) >= 0.15
    if not (opposite or check_long or lambda_like):
        return False
    # Global corr of a clean "/" is high; V/lambda is lower. Allow overlap with slash.
    global_corr = abs(_corr(xs.astype(float), ys.astype(float)))
    return global_corr <= 0.92


def interior_mark_class(crop: np.ndarray) -> str | None:
    """`slash`, `v_check`, or `filled` when interior geometry looks like a handwritten mark."""
    gray = _as_gray(crop)
    density = ink_density(crop)
    if density >= FILL_DENSITY:
        return "filled"
    if looks_like_text_line(crop):
        return None
    if _diagonal_stroke(gray):
        return "slash"
    if _v_or_check_stroke(gray):
        return "v_check"
    return None


def classify_mark(
    crop: np.ndarray | None,
    field_id: str,
    *,
    density_tau: float = DENSITY_TAU,
    fallback_dark_ratio: float | None = None,
    fallback_candidate: bool | None = None,
) -> MarkPrediction:
    """Classify a checkbox crop as marked or empty.

    Precision-first on *interior geometry*: slash, V/check/lambda, or fill.
    Does not require the PNG edge to be a unit-test printed frame (1b pads).
    Printed corners, label glyphs, and unstructured density are unmarked.
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

    kind = interior_mark_class(crop)
    if kind is not None:
        conf = 0.94 if kind == "filled" else 0.93
        return MarkPrediction(
            field_id=field_id,
            is_marked=True,
            confidence=conf,
            ink_density=round(density, 4),
            needs_hitl=False,
            source=kind,
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

    # Interior ink that is not a mark class is printed label / mis-crop, not a tick.
    return MarkPrediction(
        field_id=field_id,
        is_marked=False,
        confidence=0.9,
        ink_density=round(density, 4),
        needs_hitl=False,
        source="density",
    )

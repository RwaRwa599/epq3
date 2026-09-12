"""Checkbox mark classifier: interior ink only (printed-frame slash is not a tick)."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from med_doc.htr.blank import residual_against_blank
from med_doc.htr.schemas import MarkPrediction

# Interior-only density. Clinic empty boxes sit well below this; a real slash is above.
DENSITY_TAU = 0.10
SLASH_MIN_DENSITY = 0.06
FILL_DENSITY = 0.62
HITL_CONFIDENCE = 0.70
INSET = 0.28
# Stamp on order.json so a stale Colab zip is obvious (order-5 still had profile expansion).
TICK_POLICY = "slash-v2"
CLINIC_CROP_MIN = 36
MAX_PLAUSIBLE_TUBE_COUNT = 4


def field_never_auto_committed(field_id: str) -> bool:
    """Profiles and body-check plans expand many LIS rows — never trust a crop alone."""
    fid = (field_id or "").strip().lower()
    return fid.startswith("profile_") or fid.startswith("body_check_plan_")


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


def _working_gray(crop: np.ndarray, blank: np.ndarray | None) -> np.ndarray:
    """Dark-on-paper gray. With a blank patch, ink is the residual (printed subtracts out)."""
    if blank is None:
        return _as_gray(crop)
    resid = residual_against_blank(crop, blank)
    return np.clip(255.0 - resid, 0.0, 255.0)


def ink_density(crop: np.ndarray, threshold: float = 140.0, blank: np.ndarray | None = None) -> float:
    """Fraction of dark pixels in the checkbox *interior*, ignoring the printed frame."""
    gray = _working_gray(crop, blank)
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
    blobs = _ink_blob_count(gray)
    if blobs < 1 or blobs > 2:
        return False
    large = min(gray.shape[:2]) >= CLINIC_CROP_MIN
    # Clinic photos: a printed frame often splits into 2 blobs that still look like "/".
    if large and blobs != 1:
        return False
    ink = _ink_mask(interior)
    dens = float(ink.mean())
    if dens < SLASH_MIN_DENSITY or dens >= FILL_DENSITY:
        return False
    ys, xs = np.where(ink)
    if len(xs) < 5 or float(xs.std()) < 1e-6 or float(ys.std()) < 1e-6:
        return False
    if float(xs.std()) > 3.2 * float(ys.std()):
        return False
    corr = abs(float(np.corrcoef(xs.astype(float), ys.astype(float))[0, 1]))
    span_x = (int(xs.max()) - int(xs.min()) + 1) / float(max(interior.shape[1], 1))
    span_y = (int(ys.max()) - int(ys.min()) + 1) / float(max(interior.shape[0], 1))
    if span_x < 0.22 or span_y < 0.22:
        return False
    if large and (span_x < 0.32 or span_y < 0.32):
        return False
    # Printed rings leak into the inset; clinic crops need a cleaner diagonal.
    annulus = _annulus_dark_frac(gray)
    if large and annulus >= 0.14:
        return False
    need = 0.80 if annulus >= 0.10 else 0.58
    if large:
        need = max(need, 0.82)
    return corr >= need


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
    # V / check: arms lean opposite. Drop loose "lambda" which fires on printed rings.
    opposite = (cl * cr) < -0.05 or (cl > 0.25 and cr < -0.15) or (cl < -0.15 and cr > 0.25)
    check_long = abs(cr) >= 0.45 and len(right) >= len(left) and opposite
    if not (opposite or check_long):
        return False
    # Global corr of a clean "/" is high; V/lambda is lower. Allow overlap with slash.
    global_corr = abs(_corr(xs.astype(float), ys.astype(float)))
    return global_corr <= 0.92


def interior_mark_class(crop: np.ndarray, blank: np.ndarray | None = None) -> str | None:
    """`slash`, `v_check`, or `filled` when interior geometry looks like a handwritten mark."""
    gray = _working_gray(crop, blank)
    density = ink_density(crop, blank=blank)
    if density >= FILL_DENSITY:
        return "filled"
    if looks_like_text_line(crop if blank is None else gray):
        return None
    if _diagonal_stroke(gray):
        return "slash"
    if _v_or_check_stroke(gray):
        return "v_check"
    return None


def mark_features(crop: np.ndarray, blank: np.ndarray | None = None) -> np.ndarray:
    """Hand-crafted geometry features for the fitted mark classifier."""
    gray = _working_gray(crop, blank)
    dens = ink_density(crop, blank=blank)
    blobs = float(_ink_blob_count(gray))
    interior = _interior(gray)
    ink = _ink_mask(interior) if interior.size else np.zeros((0, 0), dtype=bool)
    corr = 0.0
    span_x = 0.0
    span_y = 0.0
    std_ratio = 0.0
    if ink.size and float(ink.mean()) > 0:
        ys, xs = np.where(ink)
        if len(xs) >= 4 and float(xs.std()) > 1e-6 and float(ys.std()) > 1e-6:
            corr = abs(float(np.corrcoef(xs.astype(float), ys.astype(float))[0, 1]))
            ih, iw = interior.shape[:2]
            span_x = (int(xs.max()) - int(xs.min()) + 1) / float(max(iw, 1))
            span_y = (int(ys.max()) - int(ys.min()) + 1) / float(max(ih, 1))
            std_ratio = min(4.0, float(xs.std()) / max(float(ys.std()), 1e-6)) / 4.0
    text = 1.0 if looks_like_text_line(crop if blank is None else gray) else 0.0
    annulus = _annulus_dark_frac(gray)
    return np.array(
        [dens, blobs / 4.0, corr, span_x, span_y, std_ratio, text, annulus],
        dtype=np.float64,
    )


# Default logistic weights: synthetic empty / slash / V / fill / printed-corner / label.
# Clinic refits write data/labels/mark_weights.json (gitignored) or MED_DOC_MARK_WEIGHTS.
_DEFAULT_LOGREG_W = np.array(
    [6.4, 1.1, 2.8, 1.6, 1.7, -1.2, -3.5, 0.4],
    dtype=np.float64,
)
_DEFAULT_LOGREG_B = -2.35
LOGREG_W = _DEFAULT_LOGREG_W
LOGREG_B = _DEFAULT_LOGREG_B


def load_logreg_weights() -> tuple[np.ndarray, float]:
    """Prefer local clinic weights, then the committed package file, then defaults."""
    import json
    import os

    from med_doc.paths import PACKAGE_DIR, ROOT

    candidates = []
    env = os.environ.get("MED_DOC_MARK_WEIGHTS")
    if env:
        candidates.append(Path(env))
    candidates.append(ROOT / "data" / "labels" / "mark_weights.json")
    candidates.append(PACKAGE_DIR / "htr" / "mark_weights.json")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            w = np.asarray(payload["w"], dtype=np.float64)
            b = float(payload["b"])
            if w.shape == _DEFAULT_LOGREG_W.shape:
                return w, b
        except Exception:
            continue
    return _DEFAULT_LOGREG_W.copy(), float(_DEFAULT_LOGREG_B)


def save_logreg_weights(path: str | Path, w: np.ndarray, b: float) -> Path:
    """Write clinic or synthetic weights. Call reload_logreg_weights() after."""
    import json

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps({"w": [float(x) for x in np.asarray(w).tolist()], "b": float(b)}, indent=2),
        encoding="utf-8",
    )
    return dest


def reload_logreg_weights() -> tuple[np.ndarray, float]:
    global LOGREG_W, LOGREG_B
    LOGREG_W, LOGREG_B = load_logreg_weights()
    return LOGREG_W, LOGREG_B


LOGREG_W, LOGREG_B = load_logreg_weights()


def logreg_mark_prob(feat: np.ndarray, w: np.ndarray | None = None, b: float | None = None) -> float:
    if w is None:
        w = LOGREG_W
    if b is None:
        b = LOGREG_B
    z = float(np.dot(feat, w) + b)
    z = float(np.clip(z, -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def train_mark_logreg(
    X: np.ndarray,
    y: np.ndarray,
    *,
    steps: int = 600,
    lr: float = 0.35,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Fit a small logistic model; numpy only (no sklearn).

    ``sample_weight`` up-weights clinic empties (false-positive pressure) without
    needing more tick crops.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if sample_weight is None:
        sw = np.ones(len(y), dtype=np.float64)
    else:
        sw = np.asarray(sample_weight, dtype=np.float64)
        sw = sw / max(float(sw.mean()), 1e-9)
    w = np.zeros(X.shape[1], dtype=np.float64)
    b = 0.0
    n = max(len(y), 1)
    for _ in range(steps):
        z = np.clip(X @ w + b, -20.0, 20.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = (p - y) * sw
        w -= lr * (X.T @ err) / n
        b -= lr * float(err.mean())
    return w, b


def classify_mark(
    crop: np.ndarray | None,
    field_id: str,
    *,
    density_tau: float = DENSITY_TAU,
    fallback_dark_ratio: float | None = None,
    fallback_candidate: bool | None = None,
    blank: np.ndarray | None = None,
) -> MarkPrediction:
    """Classify a checkbox crop as marked or empty.

    Difference-image residual when ``blank`` is the matching template patch.
    Geometry features feed a logistic score; slash/V/fill remain the mark kinds.
    """
    if crop is None or (hasattr(crop, "size") and np.asarray(crop).size == 0):
        # Block 1 dark_ratio / is_marked_candidate is debug, not a tick.
        density = float(fallback_dark_ratio or 0.0)
        return MarkPrediction(
            field_id=field_id,
            is_marked=False,
            confidence=0.4,
            ink_density=round(density, 4),
            needs_hitl=True,
            source="missing-crop",
        )

    gray = _working_gray(crop, blank)
    density = ink_density(crop, blank=blank)
    mean = float(gray.mean()) if gray.size else 255.0
    feat = mark_features(crop, blank=blank)
    p = logreg_mark_prob(feat)
    h, w = gray.shape[:2]
    if min(h, w) > 0 and max(h, w) / float(min(h, w)) > 1.45:
        return MarkPrediction(
            field_id=field_id,
            is_marked=False,
            confidence=0.9,
            ink_density=round(density, 4),
            needs_hitl=False,
            source="nonsquare-crop",
        )

    if mean < 90.0:
        return MarkPrediction(
            field_id=field_id,
            is_marked=False,
            confidence=0.85,
            ink_density=round(density, 4),
            needs_hitl=False,
            source="shadow-crop",
        )

    if looks_like_text_line(crop if blank is None else gray):
        return MarkPrediction(
            field_id=field_id,
            is_marked=False,
            confidence=0.9,
            ink_density=round(density, 4),
            needs_hitl=False,
            source="label-crop",
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

    kind = interior_mark_class(crop, blank=blank)
    clinic = min(h, w) >= CLINIC_CROP_MIN
    annulus = _annulus_dark_frac(gray)
    # Clinic photos (order-5): printed rings look like a V or a faint slash.
    # Commit only a clean `/` (or a truly filled box). V → HiTL, not LIS.
    if kind == "v_check":
        return MarkPrediction(
            field_id=field_id,
            is_marked=False,
            confidence=0.55,
            ink_density=round(density, 4),
            needs_hitl=True,
            source="v_check",
        )
    marked = False
    slash_tau = 0.11 if clinic else SLASH_MIN_DENSITY
    if kind == "slash" and density >= slash_tau:
        marked = True
        if clinic and annulus >= 0.12:
            marked = False
        if blank is not None and density < (0.12 if clinic else 0.09):
            marked = False
    fill_tau = 0.70 if clinic else FILL_DENSITY
    if kind == "filled" and density >= fill_tau:
        marked = True

    if marked and field_never_auto_committed(field_id):
        return MarkPrediction(
            field_id=field_id,
            is_marked=True,
            confidence=0.55,
            ink_density=round(density, 4),
            needs_hitl=True,
            source=kind or "panel-hitl",
        )

    if marked:
        conf = 0.94 if kind == "filled" else (0.93 if kind else 0.8)
        return MarkPrediction(
            field_id=field_id,
            is_marked=True,
            confidence=round(float(conf), 3),
            ink_density=round(density, 4),
            needs_hitl=False,
            source=kind or "logreg",
        )

    return MarkPrediction(
        field_id=field_id,
        is_marked=False,
        confidence=round(float(max(0.55, 1.0 - p)), 3),
        ink_density=round(density, 4),
        needs_hitl=p >= 0.55 and density >= SLASH_MIN_DENSITY,
        source="logreg" if p >= 0.35 else "density",
    )

"""Landmark geometry matching and column-wise fine registration."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from med_doc.normalization.detect import detect_checkboxes_photo
from med_doc.normalization.illumination import flatten_gray, ink_mask, paper_level, paper_map
from med_doc.normalization.register import piecewise_register
from med_doc.schemas import FieldSpec, TemplateSpec


def _rel_to_px(bbox: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    x0 = int(round(bbox[0] * w))
    y0 = int(round(bbox[1] * h))
    x1 = int(round(bbox[2] * w))
    y1 = int(round(bbox[3] * h))
    return x0, y0, x1, y1


def _clamp_window(
    x0: int, y0: int, x1: int, y1: int, w: int, h: int
) -> tuple[int, int, int, int]:
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    return x0, y0, x1, y1


def match_landmark(
    gray: np.ndarray,
    bbox: list[float],
    search_frac: float = 0.04,
    mode: str = "header_bar",
) -> tuple[float, float, float]:
    """Match a geometric landmark. Returns (dx, dy, score).

    The template is a synthetic edge pattern (not a crop of the page itself),
    so the match can actually move when the printed header/footer is offset.
    """
    h, w = gray.shape
    x0, y0, x1, y1 = _clamp_window(*_rel_to_px(bbox, w, h), w, h)
    th, tw = max(4, y1 - y0), max(4, x1 - x0)
    templ = np.full((th, tw), 255, dtype=np.uint8)
    if mode == "header_bar":
        templ[th // 3 : 2 * th // 3, :] = 40
    elif mode == "footer_rule":
        templ[th // 2 - 1 : th // 2 + 2, :] = 40
    else:
        templ[:, tw // 2 - 1 : tw // 2 + 2] = 40

    sx = max(8, int(round(search_frac * w)))
    sy = max(8, int(round(search_frac * h)))
    wx0, wy0, wx1, wy1 = _clamp_window(x0 - sx, y0 - sy, x1 + sx, y1 + sy, w, h)
    window = gray[wy0:wy1, wx0:wx1]
    if window.shape[0] <= th or window.shape[1] <= tw:
        return 0.0, 0.0, 0.0
    result = cv2.matchTemplate(window, templ, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    dx = float((wx0 + max_loc[0]) - x0)
    dy = float((wy0 + max_loc[1]) - y0)
    return dx, dy, float(max_val)


def _horizontal_band_shift(
    gray: np.ndarray, bbox: list[float], search: int
) -> tuple[float, float]:
    h, w = gray.shape
    x0, y0, x1, y1 = _clamp_window(*_rel_to_px(bbox, w, h), w, h)
    band_h = y1 - y0
    y_lo = max(0, y0 - search)
    y_hi = min(h, y1 + search)
    region = gray[y_lo:y_hi, x0:x1]
    edges = np.abs(cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3))
    profile = edges.mean(axis=1)
    if profile.size < band_h + 1:
        return 0.0, 0.0
    kernel = np.ones(band_h, dtype=np.float32) / max(band_h, 1)
    energy = np.convolve(profile, kernel, mode="valid")
    peak = int(np.argmax(energy))
    dy = float((y_lo + peak) - y0)
    score = float(energy[peak] / (float(energy.mean()) + 1e-6))
    return dy, min(1.0, score / 3.0)


def _column_x_peaks(gray: np.ndarray, n: int = 4) -> list[float]:
    h, w = gray.shape
    body = flatten_gray(gray[int(h * 0.10) : int(h * 0.82)])
    ink = ink_mask(body).astype(np.float32)
    proj = ink.mean(axis=0)
    k = max(5, w // 100)
    smooth = np.convolve(proj, np.ones(k) / k, mode="same")
    peaks: list[int] = []
    work = smooth.copy()
    min_sep = w // 8
    for _ in range(n):
        idx = int(np.argmax(work))
        peaks.append(idx)
        lo, hi = max(0, idx - min_sep), min(len(work), idx + min_sep)
        work[lo:hi] = 0
    return [float(x) for x in sorted(peaks)]


def _column_index(spec_x: float, cuts: np.ndarray) -> int:
    return int(np.searchsorted(cuts, spec_x))


def column_y_shifts(
    gray: np.ndarray,
    template: TemplateSpec,
    search_px: int = 24,
) -> dict[int, float]:
    h, w = gray.shape
    boxes = template.checkbox_fields()
    if not boxes:
        return {}
    xs = [0.5 * (f.bbox[0] + f.bbox[2]) for f in boxes]
    cuts = np.quantile(xs, [0.25, 0.5, 0.75])
    shifts: dict[int, float] = {}
    for col in range(4):
        members = [
            f for f in boxes if _column_index(0.5 * (f.bbox[0] + f.bbox[2]), cuts) == col
        ]
        if len(members) < 4:
            continue
        x0 = max(0, int(min(f.bbox[0] for f in members) * w) - 8)
        x1 = min(w, int(max(f.bbox[2] for f in members) * w) + 8)
        strip = flatten_gray(gray[:, x0:x1])
        ink = ink_mask(strip).mean(axis=1)
        expected = np.zeros(h, dtype=np.float32)
        for f in members:
            cy = int(round(0.5 * (f.bbox[1] + f.bbox[3]) * h))
            if 0 <= cy < h:
                expected[max(0, cy - 2) : min(h, cy + 3)] = 1.0
        if expected.sum() < 1:
            continue
        best_dy = 0
        best = -1e9
        for dy in range(-search_px, search_px + 1):
            score = float(np.dot(np.roll(ink, dy), expected))
            if score > best:
                best = score
                best_dy = dy
        shifts[col] = float(best_dy)
    return shifts


def _checkbox_grid_score(gray: np.ndarray, template: TemplateSpec) -> float:
    """Fraction of expected checkbox windows that contain a bright-interior gray ring."""
    h, w = gray.shape
    hits = 0
    n = 0
    for spec in template.checkbox_fields():
        x0, y0, x1, y1 = (
            int(round(spec.bbox[0] * w)),
            int(round(spec.bbox[1] * h)),
            int(round(spec.bbox[2] * w)),
            int(round(spec.bbox[3] * h)),
        )
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        crop = gray[y0:y1, x0:x1]
        if crop.size < 16:
            continue
        n += 1
        cy, cx = crop.shape[0] // 2, crop.shape[1] // 2
        center = float(crop[max(0, cy - 1) : cy + 2, max(0, cx - 1) : cx + 2].mean())
        border = np.concatenate([crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]])
        border_mean = float(border.mean())
        paper = paper_level(crop)
        dark = float((crop < paper * 0.55).mean())
        # Checkbox signature: center brighter than border, or clear ring
        if (center > paper * 0.72 and 0.03 <= dark <= 0.70) or (
            center - border_mean >= 6.0 and center >= paper * 0.55
        ):
            hits += 1
    return hits / max(n, 1)


def fine_align(
    canvas: np.ndarray,
    template: TemplateSpec,
) -> tuple[np.ndarray, dict[str, Any], dict[int, float]]:
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY) if canvas.ndim == 3 else canvas
    h, w = gray.shape
    before = _checkbox_grid_score(gray, template)
    dxs: list[float] = []
    dys: list[float] = []
    scores: list[float] = []

    for landmark in template.landmarks:
        dx, dy, ncc = match_landmark(gray, landmark.bbox, mode=landmark.kind)
        if ncc > 0.1:
            dxs.append(dx)
            dys.append(dy)
            scores.append(ncc)
        if landmark.kind in {"header_bar", "footer_rule"}:
            band_dy, band_score = _horizontal_band_shift(
                gray, landmark.bbox, search=max(12, h // 30)
            )
            dys.append(band_dy)
            scores.append(band_score)

    tx = float(np.median(dxs)) if dxs else 0.0
    ty = float(np.median(dys)) if dys else 0.0

    boxes = template.checkbox_fields()
    if boxes:
        xs = [0.5 * (f.bbox[0] + f.bbox[2]) * w for f in boxes]
        expected_x = [float(v) for v in np.quantile(xs, [0.125, 0.375, 0.625, 0.875])]
        found = _column_x_peaks(gray, n=4)
        if len(found) == 4:
            tx = float(np.median([f - e for f, e in zip(found, expected_x)]))

    tx = float(np.clip(tx, -w * 0.04, w * 0.04))
    ty = float(np.clip(ty, -h * 0.04, h * 0.04))

    matrix = np.array([[1.0, 0.0, -tx], [0.0, 1.0, -ty]], dtype=np.float32)
    aligned = cv2.warpAffine(
        canvas, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
    )
    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY) if aligned.ndim == 3 else aligned
    col_shifts = column_y_shifts(aligned_gray, template)
    after = _checkbox_grid_score(aligned_gray, template)
    # Only keep a shift that clearly improves checkbox registration.
    if after < before + 0.05:
        aligned = canvas
        col_shifts = {}
        tx, ty = 0.0, 0.0
        after = before
        aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY) if aligned.ndim == 3 else aligned

    piecewise_meta: dict[str, Any] = {}
    affine = None
    refined, piecewise_meta, affine = piecewise_register(aligned, template)
    refined_gray = cv2.cvtColor(refined, cv2.COLOR_RGB2GRAY) if refined.ndim == 3 else refined
    piecewise_score = _checkbox_grid_score(refined_gray, template)
    piecewise_meta["grid_score"] = round(float(piecewise_score), 4)
    if piecewise_meta.get("applied") and piecewise_score >= after + 0.02:
        aligned = refined
        after = piecewise_score
        col_shifts = column_y_shifts(refined_gray, template)
    else:
        piecewise_meta["applied"] = False

    ncc = float(np.mean(scores)) if scores else 0.5
    col_mag = float(np.mean([abs(v) for v in col_shifts.values()])) if col_shifts else 0.0
    confidence = float(
        np.clip(0.5 * max(after, ncc) + 0.5 * max(0.0, 1.0 - col_mag / 20.0), 0.0, 1.0)
    )
    meta = {
        "tx": tx,
        "ty": ty,
        "landmark_score": ncc,
        "grid_score": after,
        "column_shifts": {str(k): v for k, v in col_shifts.items()},
        "confidence": confidence,
        "piecewise": piecewise_meta,
        "affine": None if affine is None else [float(v) for v in affine.ravel()],
    }
    return aligned, meta, col_shifts


def _px_to_rel(bbox: list[int], w: int, h: int) -> list[float]:
    x0, y0, x1, y1 = (int(v) for v in bbox)
    rel = [
        max(0.0, min(1.0, x0 / w)),
        max(0.0, min(1.0, y0 / h)),
        max(0.0, min(1.0, x1 / w)),
        max(0.0, min(1.0, y1 / h)),
    ]
    if rel[2] <= rel[0]:
        rel[2] = min(1.0, rel[0] + 1.0 / w)
    if rel[3] <= rel[1]:
        rel[3] = min(1.0, rel[1] + 1.0 / h)
    return rel


# Printed black section bars, grouped by the nearest checkbox-gutter column.
# Subheads such as "Diabetes" are bold text, not bars, and are omitted.
_SECTION_BAR_GROUPS: tuple[tuple[str, ...], ...] = (
    ("haematology", "bone_nutrition", "cardiovascular"),
    ("diabetes",),
    ("immunology", "endocrinology"),
    ("tumour_markers", "urine", "stool"),
)


def _gutter_centers(template: TemplateSpec) -> list[float]:
    guts = [lm for lm in template.landmarks if lm.kind == "column_gutter"]
    guts = sorted(guts, key=lambda g: g.bbox[0])
    return [0.5 * (g.bbox[0] + g.bbox[2]) for g in guts]


def _column_of(spec: FieldSpec, gutters: list[float]) -> int:
    cx = 0.5 * (spec.bbox[0] + spec.bbox[2])
    if not gutters:
        return 0
    return int(np.argmin([abs(cx - gx) for gx in gutters]))


def _column_bounds(gutters: list[float]) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for i, x in enumerate(gutters):
        left = max(0.0, x - 0.018)
        right = (gutters[i + 1] - 0.018) if i + 1 < len(gutters) else 0.995
        bounds.append((left, right))
    return bounds


def _detect_section_bars(
    gray: np.ndarray, x0: int, x1: int, y_min: float, y_max: float
) -> list[float]:
    """Y centers of full-width dark header bars in a column strip."""
    strip = gray[:, x0:x1]
    if strip.size == 0:
        return []
    pmap = paper_map(strip, tile=48)
    row_dark = (strip < pmap * 0.55).mean(axis=1)
    row_mean = strip.mean(axis=1)
    bars: list[float] = []
    in_bar = False
    y0 = 0
    height = strip.shape[0]
    for y in range(height):
        paper = float(pmap[y].mean())
        is_bar = bool(row_dark[y] > 0.48 and row_mean[y] < paper * 0.62)
        if is_bar and not in_bar:
            in_bar = True
            y0 = y
        elif not is_bar and in_bar:
            in_bar = False
            cy = (y0 + y) / 2.0
            if 8 <= (y - y0) <= 50 and y_min < cy < y_max:
                bars.append(cy)
    return bars


def _expected_section_bars(
    members: list[FieldSpec], groups: tuple[str, ...], height: int, y_min: float
) -> list[float]:
    expected: list[float] = []
    for group in groups:
        ys = [f.bbox[1] for f in members if (f.group or "") == group]
        if not ys:
            continue
        y = float(min(ys) * height - 26)
        if y > y_min:
            expected.append(y)
    return expected


def _fit_affine_y(expected: np.ndarray, detected: np.ndarray) -> tuple[float, float, float]:
    design = np.vstack([expected, np.ones_like(expected)]).T
    scale, shift = np.linalg.lstsq(design, detected, rcond=None)[0]
    if not (0.90 <= float(scale) <= 1.10):
        scale = 1.0
        shift = float(np.median(detected - expected))
    residual = float(np.max(np.abs(scale * expected + shift - detected)))
    return float(scale), float(shift), residual


def _pair_section_bars(
    expected: list[float], detected: list[float]
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(expected) < 2 or len(detected) < 2:
        return None
    exp = np.array(sorted(expected), dtype=float)
    det = np.array(sorted(detected), dtype=float)
    if len(det) == len(exp):
        return exp, det
    best: tuple[np.ndarray, np.ndarray] | None = None
    best_res = 1e9
    if len(det) > len(exp):
        width = len(exp)
        source, target = det, exp
        extra_on_det = True
    else:
        width = len(det)
        source, target = exp, det
        extra_on_det = False
    for i in range(len(source) - width + 1):
        window = source[i : i + width]
        if extra_on_det:
            pair_e, pair_d = target, window
        else:
            pair_e, pair_d = window, target
        _scale, _shift, residual = _fit_affine_y(pair_e, pair_d)
        if residual < best_res:
            best_res = residual
            best = (pair_e, pair_d)
    return best


def _apply_column_affine(
    template: TemplateSpec,
    col_ab: dict[int, tuple[float, float]],
    gutters: list[float],
) -> TemplateSpec:
    w, h = template.width, template.height
    moved: list[FieldSpec] = []
    for spec in template.fields:
        col = _column_of(spec, gutters)
        if col not in col_ab:
            moved.append(spec)
            continue
        scale, shift = col_ab[col]
        x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
        cy = 0.5 * (y0 + y1)
        dy = float(np.clip(scale * cy + shift - cy, -h * 0.12, h * 0.12))
        moved.append(
            spec.model_copy(
                update={
                    "bbox": _px_to_rel(
                        [x0, int(round(y0 + dy)), x1, int(round(y1 + dy))], w, h
                    )
                }
            )
        )
    return template.model_copy(update={"fields": moved})


def _snap_rings_to_fields(
    canvas: np.ndarray,
    template: TemplateSpec,
    detected: list[list[int]],
    *,
    snap_px: float,
) -> tuple[TemplateSpec, int]:
    h, w = canvas.shape[:2]
    centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, b) for b in detected]
    used: set[int] = set()
    fields: list[FieldSpec] = []
    n_snapped = 0
    for spec in template.fields:
        if spec.field_type != "checkbox":
            fields.append(spec)
            continue
        x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        best_idx = None
        best_d = snap_px
        for idx, (px, py, _box) in enumerate(centers):
            if idx in used or abs(px - cx) > 28:
                continue
            dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if dist < best_d:
                best_d = dist
                best_idx = idx
        if best_idx is not None:
            used.add(best_idx)
            n_snapped += 1
            fields.append(
                spec.model_copy(update={"bbox": _px_to_rel(centers[best_idx][2], w, h)})
            )
        else:
            fields.append(spec)
    return template.model_copy(update={"fields": fields}), n_snapped


def _snap_overlay_sections(
    canvas: np.ndarray,
    template: TemplateSpec,
    detected: list[list[int]],
    *,
    snap_px: float,
) -> tuple[TemplateSpec, dict[str, Any]]:
    """Align columns using printed section bars, then snap one square per row."""
    h, w = canvas.shape[:2]
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY) if canvas.ndim == 3 else canvas
    gutters = _gutter_centers(template)
    meta: dict[str, Any] = {
        "method": "section-bars",
        "n_snapped": 0,
        "n_columns_fitted": 0,
        "dx": 0.0,
        "dy": 0.0,
    }
    if len(gutters) < 4:
        return template, meta
    bounds = _column_bounds(gutters)
    y_min, y_max = 0.12 * h, 0.92 * h
    col_ab: dict[int, tuple[float, float]] = {}
    columns_meta: list[dict[str, Any]] = []
    boxes = template.checkbox_fields()
    for col, (left, right) in enumerate(bounds):
        members = [f for f in boxes if _column_of(f, gutters) == col]
        groups = _SECTION_BAR_GROUPS[col] if col < len(_SECTION_BAR_GROUPS) else ()
        px0, px1 = max(0, int(left * w)), min(w, int(right * w))
        bars = _detect_section_bars(gray, px0, px1, y_min, y_max)
        expected = _expected_section_bars(members, groups, h, y_min)
        paired = _pair_section_bars(expected, bars)
        entry: dict[str, Any] = {
            "column": col,
            "n_bars": len(bars),
            "n_expected": len(expected),
        }
        if paired is None:
            columns_meta.append(entry)
            continue
        exp, det = paired
        scale, shift, residual = _fit_affine_y(exp, det)
        entry.update({"scale": round(scale, 3), "shift": round(shift, 1), "residual": round(residual, 1)})
        if residual <= 18.0 and abs(shift) <= 0.10 * h and len(exp) >= 2:
            col_ab[col] = (scale, shift)
            entry["fitted"] = True
        columns_meta.append(entry)
    meta["columns"] = columns_meta
    meta["n_columns_fitted"] = len(col_ab)
    if len(col_ab) < 1:
        return template, meta
    shifted = _apply_column_affine(template, col_ab, gutters)
    snapped, n_snapped = _snap_rings_to_fields(canvas, shifted, detected, snap_px=snap_px)
    meta["n_snapped"] = n_snapped
    meta["n_offset_hits"] = n_snapped
    meta["confidence"] = float(np.clip(n_snapped / max(len(shifted.checkbox_fields()), 1), 0.0, 1.0))
    return snapped, meta


def snap_overlay(
    canvas: np.ndarray,
    template: TemplateSpec,
    *,
    search_px: float = 64.0,
    snap_px: float = 30.0,
) -> tuple[TemplateSpec, dict[str, Any]]:
    """Snap the overlay using hollow rings, or section bars if they score better.

    Do not fit a min/max envelope from an incomplete grid — that caused the
    ~40 px Y drift on dark clinic photos.
    """
    h, w = canvas.shape[:2]
    detected = detect_checkboxes_photo(canvas)
    meta: dict[str, Any] = {
        "n_detected": len(detected),
        "n_snapped": 0,
        "dx": 0.0,
        "dy": 0.0,
        "method": "frozen",
    }
    if len(detected) < 30:
        section, section_meta = _snap_overlay_sections(
            canvas, template, detected, snap_px=snap_px
        )
        if section_meta.get("n_columns_fitted", 0):
            section_meta["n_detected"] = len(detected)
            return section, section_meta
        return template, meta

    centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, b) for b in detected]
    offsets: list[tuple[float, float]] = []
    for spec in template.checkbox_fields():
        x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        best_d = search_px
        best = None
        for px, py, box in centers:
            d = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best = (px - cx, py - cy)
        if best is not None:
            offsets.append(best)

    dx = float(np.median([o[0] for o in offsets])) if offsets else 0.0
    dy = float(np.median([o[1] for o in offsets])) if offsets else 0.0
    dx = float(np.clip(dx, -w * 0.06, w * 0.06))
    dy = float(np.clip(dy, -h * 0.06, h * 0.06))
    meta["dx"] = round(dx, 2)
    meta["dy"] = round(dy, 2)
    meta["n_offset_hits"] = len(offsets)

    shifted: list[FieldSpec] = []
    for spec in template.fields:
        x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
        shifted.append(
            spec.model_copy(
                update={"bbox": _px_to_rel([x0 + int(round(dx)), y0 + int(round(dy)), x1 + int(round(dx)), y1 + int(round(dy))], w, h)}
            )
        )
    moved = template.model_copy(update={"fields": shifted})

    used_boxes: set[int] = set()
    snapped_fields: list[FieldSpec] = []
    n_snapped = 0
    for spec in moved.fields:
        if spec.field_type != "checkbox":
            snapped_fields.append(spec)
            continue
        x0, y0, x1, y1 = moved.pixel_bbox(spec, apply_pad=False)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        best_idx = None
        best_d = snap_px
        for idx, (px, py, box) in enumerate(centers):
            if idx in used_boxes:
                continue
            if abs(px - cx) > 28:
                continue
            d = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best_idx = idx
        if best_idx is not None:
            used_boxes.add(best_idx)
            n_snapped += 1
            snapped_fields.append(spec.model_copy(update={"bbox": _px_to_rel(centers[best_idx][2], w, h)}))
        else:
            snapped_fields.append(spec)
    meta["n_snapped"] = n_snapped
    meta["method"] = "global-shift-then-snap"
    meta["confidence"] = float(np.clip(n_snapped / max(len(moved.checkbox_fields()), 1), 0.0, 1.0))
    rings = moved.model_copy(update={"fields": snapped_fields})

    section, section_meta = _snap_overlay_sections(
        canvas, template, detected, snap_px=snap_px
    )
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY) if canvas.ndim == 3 else canvas
    ring_score = _checkbox_grid_score(gray, rings)
    section_score = _checkbox_grid_score(gray, section)
    meta["grid_score"] = round(ring_score, 4)
    meta["section_grid_score"] = round(section_score, 4)
    # Keep section-bar snap only when it clearly improves hollow-grid registration.
    if section_meta.get("n_columns_fitted", 0) and section_score >= ring_score + 0.02:
        section_meta["n_detected"] = len(detected)
        section_meta["grid_score"] = round(section_score, 4)
        section_meta["ring_grid_score"] = round(ring_score, 4)
        return section, section_meta
    return rings, meta

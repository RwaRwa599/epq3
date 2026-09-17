"""if1block1 layout warp from CHECK-UP / section bars and column gutters.

Not a paper-quad homography. Too few landmarks → ``warp_to_canonical`` fallback.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from med_doc.normalization.align import (
    _SECTION_BAR_GROUPS,
    _column_bounds,
    _column_of,
    _column_x_peaks,
    _expected_section_bars,
    _gutter_centers,
    detect_checkup_header_bar,
)
from med_doc.normalization.sections import detect_header_bars
from med_doc.normalization.warp import load_rgb, warp_to_canonical
from med_doc.schemas import TemplateSpec

_MIN_POINTS = 3


def _as_gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _header_landmark(template: TemplateSpec):
    for landmark in template.landmarks:
        if landmark.kind == "header_bar" or landmark.id == "header_bar":
            return landmark
    return None


def _collect_correspondences(
    gray: np.ndarray,
    template: TemplateSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Src (image px) ↔ dst (canonical template px)."""
    h, w = gray.shape[:2]
    dw, dh = template.width, template.height
    src: list[list[float]] = []
    dst: list[list[float]] = []
    meta: dict[str, Any] = {"n_header": 0, "n_section_bars": 0, "n_gutters": 0}

    header = detect_checkup_header_bar(gray)
    lm = _header_landmark(template)
    if float(header.get("score") or 0.0) >= 0.18 and lm is not None:
        cy_t = 0.5 * (lm.bbox[1] + lm.bbox[3]) * dh
        x0 = float(header.get("x0") or int(0.02 * w))
        x1 = float(header.get("x1") or int(0.98 * w))
        cy = float(header["cy"])
        src.extend([[x0, cy], [x1, cy]])
        dst.extend([[lm.bbox[0] * dw, cy_t], [lm.bbox[2] * dw, cy_t]])
        meta["n_header"] = 1

    boxes = template.checkbox_fields()
    gutters = _gutter_centers(template)
    bounds = _column_bounds(gutters) if gutters else [(0.02, 0.98)]
    y_min, y_max = 0.0, 0.92 * h
    n_bars = 0
    for col, (left, right) in enumerate(bounds):
        px0, px1 = max(0, int(left * w)), min(w, int(right * w))
        if px1 - px0 < 16:
            continue
        bars = detect_header_bars(gray, px0, px1, y_min=y_min, y_max=y_max)
        members = [f for f in boxes if gutters and _column_of(f, gutters) == col] or boxes
        groups = _SECTION_BAR_GROUPS[col] if col < len(_SECTION_BAR_GROUPS) else ()
        expected = _expected_section_bars(members, groups, dh, 0.0)
        detected_cy = [0.5 * (b[1] + b[3]) for b in bars]
        if not expected or not detected_cy:
            continue
        exp = sorted(expected)
        det = sorted(detected_cy)
        n = min(len(exp), len(det))
        by_cy = {0.5 * (b[1] + b[3]): b for b in bars}
        for ey, dy in zip(exp[:n], det[:n]):
            bar = min(by_cy, key=lambda c: abs(c - dy), default=None)
            if bar is None:
                continue
            bx0, _by0, bx1, _by1 = by_cy[bar]
            src.extend([[float(bx0), float(dy)], [float(bx1), float(dy)]])
            dst.extend([[left * dw, float(ey)], [right * dw, float(ey)]])
            n_bars += 1
    meta["n_section_bars"] = n_bars

    found = _column_x_peaks(gray, n=4)
    if len(found) == 4 and boxes:
        xs = [0.5 * (f.bbox[0] + f.bbox[2]) * dw for f in boxes]
        expected_x = [float(v) for v in np.quantile(xs, [0.125, 0.375, 0.625, 0.875])]
        y_src = 0.45 * h
        y_dst = 0.45 * dh
        for fs, es in zip(found, expected_x):
            src.append([float(fs), y_src])
            dst.append([es, y_dst])
        meta["n_gutters"] = 4
    elif gutters:
        found_n = _column_x_peaks(gray, n=max(len(gutters), 1))
        y_src = 0.45 * h
        y_dst = 0.45 * dh
        for gx_img, gx_t in zip(found_n, gutters):
            src.append([float(gx_img), y_src])
            dst.append([gx_t * dw, y_dst])
        meta["n_gutters"] = min(len(found_n), len(gutters))

    if not src:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), meta
    return (
        np.asarray(src, dtype=np.float32),
        np.asarray(dst, dtype=np.float32),
        meta,
    )


def layout_warp(
    image: Image.Image | np.ndarray | str,
    template: TemplateSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Warp a photo onto the template canvas using bar/gutter correspondences."""
    rgb = load_rgb(image)
    gray = _as_gray(rgb)
    dest = (template.width, template.height)
    src, dst, geom = _collect_correspondences(gray, template)
    meta: dict[str, Any] = {"layout": geom, "padded": False}
    if len(src) < _MIN_POINTS:
        canvas, fallback = warp_to_canonical(rgb, dest_size=dest)
        meta.update(fallback)
        meta["method"] = str(fallback.get("method") or "full-frame")
        meta["layout_fallback"] = "too_few_landmarks"
        return canvas, meta

    affine, inliers = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=8.0
    )
    n_in = int(inliers.sum()) if inliers is not None else 0
    if affine is None or n_in < _MIN_POINTS:
        canvas, fallback = warp_to_canonical(rgb, dest_size=dest)
        meta.update(fallback)
        meta["method"] = str(fallback.get("method") or "full-frame")
        meta["layout_fallback"] = "affine_failed"
        meta["n_inliers"] = n_in
        return canvas, meta

    warped = cv2.warpAffine(
        rgb,
        affine,
        dest,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    # Letterbox: if the mapped source bbox is inset, warp already filled white.
    mapped = cv2.transform(src.reshape(-1, 1, 2), affine).reshape(-1, 2)
    xs, ys = mapped[:, 0], mapped[:, 1]
    inset = bool(xs.min() > 8 or ys.min() > 8 or xs.max() < dest[0] - 8 or ys.max() < dest[1] - 8)
    meta.update(
        {
            "method": "bar-gutter-affine",
            "n_points": int(len(src)),
            "n_inliers": n_in,
            "padded": inset,
            "affine": [float(v) for v in affine.ravel()],
            "confidence": float(np.clip(n_in / max(len(src), 1), 0.0, 1.0)),
            "orientation_degrees": 0,
        }
    )
    return warped, meta

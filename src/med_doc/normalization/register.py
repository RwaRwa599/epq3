"""RANSAC / piecewise page registration used by Block 1a and 1c rematch."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from med_doc.normalization.detect import detect_checkboxes_photo, filter_checkbox_boxes
from med_doc.normalization.illumination import flatten_gray
from med_doc.schemas import TemplateSpec


def ransac_partial_affine(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    n_iter: int = 96,
    thresh: float = 10.0,
    min_inliers: int = 8,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Fit a partial affine (scale, rotation, translation) with RANSAC.

    ``src`` and ``dst`` are Nx2. Returns (2x3 matrix or None, inlier mask).
    """
    src = np.asarray(src, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float32).reshape(-1, 2)
    n = int(src.shape[0])
    if n < 3 or dst.shape[0] != n:
        return None, np.zeros(n, dtype=bool)
    rng = rng or np.random.default_rng(0)
    best_mask = np.zeros(n, dtype=bool)
    best_n = 0
    for _ in range(n_iter):
        idx = rng.choice(n, size=3, replace=False)
        m, _ = cv2.estimateAffinePartial2D(
            src[idx], dst[idx], method=cv2.LMEDS, refineIters=0
        )
        if m is None:
            continue
        pred = apply_affine(m, src)
        err = np.linalg.norm(pred - dst, axis=1)
        mask = err < thresh
        count = int(mask.sum())
        if count > best_n:
            best_n = count
            best_mask = mask
    if best_n < min(min_inliers, max(3, n // 4)):
        # Translation-only fallback from the median offset.
        med = np.median(dst - src, axis=0)
        m = np.array([[1.0, 0.0, float(med[0])], [0.0, 1.0, float(med[1])]], dtype=np.float32)
        pred = apply_affine(m, src)
        mask = np.linalg.norm(pred - dst, axis=1) < thresh
        return m, mask
    m, _ = cv2.estimateAffinePartial2D(src[best_mask], dst[best_mask], method=cv2.LMEDS)
    if m is None:
        med = np.median(dst[best_mask] - src[best_mask], axis=0)
        m = np.array([[1.0, 0.0, float(med[0])], [0.0, 1.0, float(med[1])]], dtype=np.float32)
    return m.astype(np.float32), best_mask


def apply_affine(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    hom = np.hstack([pts, ones])
    return (m.astype(np.float32) @ hom.T).T


def predicted_center(m: np.ndarray | None, xy: tuple[float, float]) -> tuple[float, float]:
    if m is None:
        return xy
    out = apply_affine(m, np.array([xy], dtype=np.float32))[0]
    return float(out[0]), float(out[1])


def _expected_centers(template: TemplateSpec) -> tuple[np.ndarray, list[str]]:
    pts: list[list[float]] = []
    ids: list[str] = []
    for spec in template.checkbox_fields():
        x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
        pts.append([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
        ids.append(spec.field_id)
    return np.asarray(pts, dtype=np.float32), ids


def match_expected_to_detected(
    expected: np.ndarray,
    detected: list[list[int]],
    *,
    max_dist: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy nearest-neighbour pairing. Returns (src expected, dst detected) Nx2."""
    if expected.size == 0 or not detected:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    det = np.array(
        [[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in detected], dtype=np.float32
    )
    used: set[int] = set()
    src: list[list[float]] = []
    dst: list[list[float]] = []
    for p in expected:
        best_i = -1
        best_d = max_dist
        for i, q in enumerate(det):
            if i in used:
                continue
            d = float(np.hypot(float(q[0] - p[0]), float(q[1] - p[1])))
            if d < best_d:
                best_d = d
                best_i = i
        if best_i >= 0:
            used.add(best_i)
            src.append([float(p[0]), float(p[1])])
            dst.append([float(det[best_i, 0]), float(det[best_i, 1])])
    if not src:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    return np.asarray(src, np.float32), np.asarray(dst, np.float32)


def _bilinear_flow(h: int, w: int, samples: np.ndarray, residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate residual (dx, dy) from sparse sample points onto a dense map."""
    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)
    if len(samples) == 0:
        return dx, dy
    if len(samples) == 1:
        dx[:] = residuals[0, 0]
        dy[:] = residuals[0, 1]
        return dx, dy
    # Coarse grid of medians, then resize.
    gy, gx = 3, 4
    cell_h, cell_w = h / gy, w / gx
    gdx = np.zeros((gy, gx), dtype=np.float32)
    gdy = np.zeros((gy, gx), dtype=np.float32)
    gw = np.zeros((gy, gx), dtype=np.float32)
    for (x, y), (rx, ry) in zip(samples, residuals):
        ix = int(np.clip(x / cell_w, 0, gx - 1))
        iy = int(np.clip(y / cell_h, 0, gy - 1))
        gdx[iy, ix] += rx
        gdy[iy, ix] += ry
        gw[iy, ix] += 1.0
    filled = gw > 0
    if not np.any(filled):
        return dx, dy
    # Fill empty cells with global median, then neighbour average.
    mx, my = float(np.median(residuals[:, 0])), float(np.median(residuals[:, 1]))
    gdx = np.where(filled, gdx / np.maximum(gw, 1.0), mx)
    gdy = np.where(filled, gdy / np.maximum(gw, 1.0), my)
    dx = cv2.resize(gdx, (w, h), interpolation=cv2.INTER_LINEAR)
    dy = cv2.resize(gdy, (w, h), interpolation=cv2.INTER_LINEAR)
    return dx, dy


def _ecc_edges(gray: np.ndarray) -> np.ndarray:
    work = flatten_gray(gray) if gray.ndim == 2 else flatten_gray(cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY))
    work = cv2.GaussianBlur(work, (5, 5), 0)
    gx = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    p = float(np.percentile(mag, 98)) or 1.0
    return np.clip(mag * (255.0 / p), 0, 255).astype(np.float32)


def ecc_refine(
    canvas: np.ndarray,
    template: TemplateSpec,
    *,
    max_side: int = 800,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Euclidean ECC of the page onto a rendered blank template (no PHI)."""
    from med_doc.htr.blank import render_blank_form

    meta: dict[str, Any] = {"applied": False, "cc": 0.0, "method": "ecc_euclidean"}
    h, w = canvas.shape[:2]
    blank = render_blank_form(template, labels=False)
    if blank.shape[0] != h or blank.shape[1] != w:
        blank = cv2.resize(blank, (w, h), interpolation=cv2.INTER_AREA)
    scale = min(1.0, float(max_side) / float(max(h, w)))
    sw, sh = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
    page_e = _ecc_edges(canvas)
    blank_e = _ecc_edges(blank)
    if scale < 0.999:
        page_s = cv2.resize(page_e, (sw, sh), interpolation=cv2.INTER_AREA)
        blank_s = cv2.resize(blank_e, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        page_s, blank_s = page_e, blank_e
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    try:
        try:
            cc, warp = cv2.findTransformECC(
                blank_s, page_s, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5
            )
        except TypeError:
            cc, warp = cv2.findTransformECC(
                blank_s, page_s, warp, cv2.MOTION_EUCLIDEAN, criteria
            )
    except cv2.error:
        meta["skipped"] = "ecc_failed"
        return canvas, meta
    meta["cc"] = round(float(cc), 4)
    if scale < 0.999:
        warp = warp.copy()
        warp[0, 2] /= scale
        warp[1, 2] /= scale
    tx, ty = float(warp[0, 2]), float(warp[1, 2])
    rot = abs(float(warp[0, 0]) - 1.0) + abs(float(warp[1, 1]) - 1.0)
    if abs(tx) > 0.08 * w or abs(ty) > 0.08 * h or rot > 0.12:
        meta["skipped"] = "magnitude"
        return canvas, meta
    if float(cc) < 0.35:
        meta["skipped"] = "low_cc"
        return canvas, meta
    aligned = cv2.warpAffine(
        canvas,
        warp,
        (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderValue=(255, 255, 255),
    )
    meta["applied"] = True
    meta["warp"] = [float(v) for v in warp.ravel()]
    return aligned, meta


def neighbour_offset(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[np.ndarray | None, tuple[float, float]]:
    """RANSAC partial-affine on confirmed neighbours; returns (matrix, median dy/dx)."""
    src = np.asarray(src, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float32).reshape(-1, 2)
    if src.shape[0] < 2 or dst.shape[0] != src.shape[0]:
        med = (0.0, 0.0)
        if src.shape[0] == 1:
            med = (float(dst[0, 0] - src[0, 0]), float(dst[0, 1] - src[0, 1]))
        return None, med
    m, inn = ransac_partial_affine(src, dst, thresh=12.0, min_inliers=max(2, src.shape[0] // 3))
    if m is None:
        med = np.median(dst - src, axis=0)
        return None, (float(med[0]), float(med[1]))
    return m, (float(m[0, 2]), float(m[1, 2]))


def warp_piecewise(canvas: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    h, w = canvas.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xs + dx
    map_y = ys + dy
    border = (255, 255, 255) if canvas.ndim == 3 else 255
    return cv2.remap(
        canvas,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def piecewise_register(
    canvas: np.ndarray,
    template: TemplateSpec,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    """Local residual warp after global 1a align. Identity if too few inliers."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY) if canvas.ndim == 3 else canvas
    flat = flatten_gray(gray)
    detected = filter_checkbox_boxes(flat, detect_checkboxes_photo(flat))
    expected, _ids = _expected_centers(template)
    src, dst = match_expected_to_detected(expected, detected, max_dist=64.0)
    meta: dict[str, Any] = {
        "n_detected": len(detected),
        "n_paired": int(src.shape[0]),
        "n_inliers": 0,
        "applied": False,
    }
    if src.shape[0] < 6:
        return canvas, meta, None
    m, inn = ransac_partial_affine(src, dst, thresh=14.0, min_inliers=6)
    meta["n_inliers"] = int(inn.sum()) if inn.size else 0
    if m is None or meta["n_inliers"] < 6:
        return canvas, meta, m
    pred = apply_affine(m, src[inn])
    residual = dst[inn] - pred
    dx, dy = _bilinear_flow(gray.shape[0], gray.shape[1], pred, residual)
    # Combined warp: affine + residual. Build a full map from identity.
    h, w = gray.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    # Inverse of partial affine for sampling: dst = A @ src => src ≈ A^{-1} @ dst
    a = np.vstack([m, [0.0, 0.0, 1.0]]).astype(np.float32)
    try:
        inv = np.linalg.inv(a)[:2]
    except np.linalg.LinAlgError:
        return canvas, meta, m
    ones = np.ones((h * w, 1), dtype=np.float32)
    pts = np.hstack([xs.reshape(-1, 1), ys.reshape(-1, 1), ones])
    src_xy = (inv @ pts.T).T
    map_x = (src_xy[:, 0].reshape(h, w) + dx).astype(np.float32)
    map_y = (src_xy[:, 1].reshape(h, w) + dy).astype(np.float32)
    mag = float(np.median(np.hypot(dx[::8, ::8], dy[::8, ::8])))
    # Skip a wild warp (bad correspondences).
    if mag > 28.0 or abs(float(m[0, 2])) > 0.08 * w or abs(float(m[1, 2])) > 0.08 * h:
        meta["skipped"] = "magnitude"
        return canvas, meta, m
    border = (255, 255, 255) if canvas.ndim == 3 else 255
    aligned = cv2.remap(
        canvas,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    meta["applied"] = True
    meta["affine"] = [float(v) for v in m.ravel()]
    meta["residual_median"] = round(mag, 3)
    return aligned, meta, m

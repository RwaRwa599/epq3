"""Programmatic photo-realistic distortions of digital blanks (no PHI)."""

from __future__ import annotations

import cv2
import numpy as np


def shadow_gradient(img: np.ndarray, strength: float = 0.45, axis: int = 1) -> np.ndarray:
    """Flash/shadow ramp along x (axis=1) or y (axis=0)."""
    work = img.astype(np.float32)
    h, w = work.shape[:2]
    if axis == 1:
        lin = np.linspace(1.0 - strength, 1.0 + 0.08, w, dtype=np.float32)
        ramp = lin.reshape(1, w, 1) if work.ndim == 3 else lin.reshape(1, w)
    else:
        lin = np.linspace(1.0 - strength, 1.0 + 0.08, h, dtype=np.float32)
        ramp = lin.reshape(h, 1, 1) if work.ndim == 3 else lin.reshape(h, 1)
    return np.clip(work * ramp, 0, 255).astype(np.uint8)


def perspective_jitter(img: np.ndarray, amount: float = 0.035, seed: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    jitter = rng.uniform(-amount, amount, size=(4, 2)).astype(np.float32)
    jitter[:, 0] *= w
    jitter[:, 1] *= h
    dst = src + jitter
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), borderValue=(255, 255, 255))


def gaussian_blur(img: np.ndarray, sigma: float = 1.1) -> np.ndarray:
    k = max(3, int(sigma * 4) | 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def jpeg_artifacts(img: np.ndarray, quality: int = 42) -> np.ndarray:
    if img.ndim == 2:
        work = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        work = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", work, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if decoded is None:
        return img
    if img.ndim == 2:
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def distort_sheet(img: np.ndarray, *, seed: int = 0, strength: float = 0.4) -> np.ndarray:
    """Compose perspective + shadow + mild blur + JPEG. Deterministic per seed."""
    rng = np.random.default_rng(seed)
    out = perspective_jitter(img, amount=0.02 + 0.02 * strength, seed=seed)
    axis = int(rng.integers(0, 2))
    out = shadow_gradient(out, strength=0.25 + 0.35 * strength, axis=axis)
    out = gaussian_blur(out, sigma=0.6 + 0.8 * strength)
    q = int(rng.integers(38, 72))
    return jpeg_artifacts(out, quality=q)


def draw_ticks(page: np.ndarray, boxes: list[list[int]], kind: str = "slash") -> np.ndarray:
    out = page.copy()
    for x0, y0, x1, y1 in boxes:
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if kind == "fill":
            cv2.rectangle(out, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), (20, 20, 20), -1)
        elif kind == "v":
            cv2.line(out, (cx - 5, cy - 2), (cx, cy + 5), (18, 18, 18), 2)
            cv2.line(out, (cx, cy + 5), (cx + 7, cy - 6), (18, 18, 18), 2)
        else:
            cv2.line(out, (x0 + 4, y0 + 4), (x1 - 4, y1 - 4), (18, 18, 18), 2)
    return out

"""Locally adaptive paper / ink cuts for flash and shadow gradients."""

from __future__ import annotations

import cv2
import numpy as np

Paper = float | np.ndarray


def _as_u8(gray: np.ndarray) -> np.ndarray:
    arr = np.asarray(gray)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def paper_map(gray: np.ndarray, tile: int = 64) -> np.ndarray:
    """Per-tile paper level (percentile 90), bilinear-upsampled to the image."""
    work = _as_u8(gray)
    h, w = work.shape
    tile = max(16, int(tile))
    ny = max(1, int(np.ceil(h / tile)))
    nx = max(1, int(np.ceil(w / tile)))
    grid = np.zeros((ny, nx), dtype=np.float32)
    for iy in range(ny):
        y0, y1 = iy * tile, min(h, (iy + 1) * tile)
        for ix in range(nx):
            x0, x1 = ix * tile, min(w, (ix + 1) * tile)
            patch = work[y0:y1, x0:x1]
            if patch.size == 0:
                grid[iy, ix] = 255.0
                continue
            # Otsu brighter-class mean, else percentile — both local to the tile.
            if patch.size >= 64:
                _ret, th = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                bright = patch[th > 0]
                if bright.size >= 8:
                    grid[iy, ix] = float(np.mean(bright))
                    continue
            grid[iy, ix] = float(np.percentile(patch, 90))
    if ny == 1 and nx == 1:
        return np.full((h, w), float(grid[0, 0]), dtype=np.float32)
    return cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)


def paper_level(gray: np.ndarray) -> float:
    """Scalar paper estimate for a small crop (tile Otsu / bright-class mean)."""
    work = _as_u8(gray)
    if work.size == 0:
        return 255.0
    if work.size >= 32:
        _ret, th = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright = work[th > 0]
        if bright.size >= 8:
            return float(np.mean(bright))
    return float(np.percentile(work, 90))


def paper_at(paper: Paper, y: int, x: int) -> float:
    if isinstance(paper, np.ndarray) and paper.ndim == 2:
        y = int(np.clip(y, 0, paper.shape[0] - 1))
        x = int(np.clip(x, 0, paper.shape[1] - 1))
        return float(paper[y, x])
    return float(paper)


def flatten_gray(gray: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Background-division flatten so a global cut is closer to valid after 1a."""
    work = _as_u8(gray)
    h, w = work.shape
    if sigma is None:
        sigma = max(12.0, min(h, w) / 24.0)
    k = max(3, int(sigma) * 2 + 1)
    background = cv2.GaussianBlur(work, (k, k), sigma)
    background = np.maximum(background, 1)
    return cv2.divide(work, background, scale=255)


def ink_mask(gray: np.ndarray, paper: Paper | None = None, frac: float = 0.55) -> np.ndarray:
    work = _as_u8(gray)
    if paper is None:
        if min(work.shape) >= 48:
            paper = paper_map(work)
        else:
            paper = paper_level(work)
    if isinstance(paper, np.ndarray) and paper.shape[:2] == work.shape[:2]:
        cut = np.maximum(paper * frac, 40.0)
        return work < cut
    cut = max(40.0, float(paper) * frac)
    return work < cut

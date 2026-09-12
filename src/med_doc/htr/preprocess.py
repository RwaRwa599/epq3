"""Verbal crop prep: local paper, optional blank residual, line/char splits."""

from __future__ import annotations

import cv2
import numpy as np

from med_doc.normalization.illumination import paper_level


def as_gray(crop: np.ndarray) -> np.ndarray:
    arr = np.asarray(crop)
    if arr.ndim == 3:
        return arr.mean(axis=2).astype(np.float32)
    return arr.astype(np.float32)


def working_gray(crop: np.ndarray, blank: np.ndarray | None = None) -> np.ndarray:
    """Dark-on-paper gray; printed form subtracts away when ``blank`` is given."""
    if blank is None:
        return as_gray(crop)
    from med_doc.htr.blank import residual_against_blank

    resid = residual_against_blank(crop, blank)
    return np.clip(255.0 - resid, 0.0, 255.0)


def ink_mask(gray: np.ndarray, threshold: float = 140.0) -> np.ndarray:
    work = as_gray(gray)
    if work.size == 0:
        return np.zeros((0, 0), dtype=bool)
    paper = paper_level(work)
    cut = min(threshold, max(40.0, paper * 0.55))
    mask = work < cut
    if mask.ndim == 2 and mask.shape[0] >= 8:
        header_lim = max(2, int(mask.shape[0] * 0.18))
        row_frac = mask.mean(axis=1)
        idx = np.arange(mask.shape[0])
        mask[(idx < header_lim) & (row_frac > 0.42)] = False
    return mask


def crop_to_ink(crop: np.ndarray, pad: int = 8, blank: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(crop)
    gray = working_gray(arr, blank)
    mask = ink_mask(gray)
    ys, xs = np.where(mask)
    if len(ys) < 12:
        return arr
    h, w = gray.shape[:2]
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    if y1 - y0 < 4 or x1 - x0 < 4:
        return arr
    return arr[y0:y1, x0:x1]


def has_ink(
    crop: np.ndarray | None,
    threshold: float = 140.0,
    min_frac: float | None = None,
    blank: np.ndarray | None = None,
) -> bool:
    if crop is None:
        return False
    gray = working_gray(crop, blank)
    if gray.size == 0 or min(gray.shape[:2]) < 4:
        return False
    h, w = gray.shape[:2]
    interior = gray[int(h * 0.08) : int(h * 0.92), int(w * 0.06) : int(w * 0.94)]
    if interior.size == 0:
        interior = gray
    mask = ink_mask(interior, threshold)
    ih, iw = interior.shape[:2]
    inner = mask.copy()
    inner[: int(ih * 0.12), :] = False
    inner[int(ih * 0.88) :, :] = False
    inner[:, : int(iw * 0.10)] = False
    inner[:, int(iw * 0.90) :] = False
    n_ink = int(inner.sum())
    frac = float(n_ink) / float(max(inner.size, 1))
    if min_frac is not None:
        return frac >= min_frac
    adaptive = max(18.0 / float(interior.size), 0.005)
    return n_ink >= 18 and frac >= adaptive


def split_line_bands(mask: np.ndarray, *, min_height: int = 8) -> list[tuple[int, int]]:
    """Inclusive-exclusive Y ranges of ink rows."""
    if mask.size == 0:
        return []
    row = mask.mean(axis=1)
    on = row > 0.04
    bands: list[tuple[int, int]] = []
    y0 = None
    for y, flag in enumerate(on):
        if flag and y0 is None:
            y0 = y
        elif not flag and y0 is not None:
            if y - y0 >= min_height:
                bands.append((y0, y))
            y0 = None
    if y0 is not None and mask.shape[0] - y0 >= min_height:
        bands.append((y0, mask.shape[0]))
    if not bands and on.any():
        ys = np.where(on)[0]
        bands.append((int(ys.min()), int(ys.max()) + 1))
    return bands


def split_char_boxes(mask: np.ndarray, *, min_width: int = 3) -> list[tuple[int, int]]:
    """Inclusive-exclusive X ranges of glyph columns."""
    if mask.size == 0:
        return []
    col = mask.mean(axis=0)
    on = col > 0.05
    boxes: list[tuple[int, int]] = []
    x0 = None
    for x, flag in enumerate(on):
        if flag and x0 is None:
            x0 = x
        elif not flag and x0 is not None:
            if x - x0 >= min_width:
                boxes.append((x0, x))
            x0 = None
    if x0 is not None and mask.shape[1] - x0 >= min_width:
        boxes.append((x0, mask.shape[1]))
    return boxes

"""Checkbox detection on photographed clinic sheets (looser than the digital blank)."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from med_doc.normalization.illumination import paper_level

# Printed clinic boxes are ~18–24 px. Letter counters from detect are often ≥30 px.
CHECKBOX_SIDE_MIN = 12
CHECKBOX_SIDE_MAX = 26


def _as_gray(crop: np.ndarray) -> np.ndarray:
    arr = np.asarray(crop)
    if arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def _annulus_at(gray: np.ndarray) -> bool:
    """True when this window itself is a four-sided printed square (empty or slash)."""
    if gray.size == 0 or min(gray.shape[:2]) < 8:
        return False
    h, w = gray.shape[:2]
    if max(h, w) / float(min(h, w)) > 1.25:
        return False
    paper = float(paper_level(gray))
    # Synthetic blanks use ~150 gray rings; clinic ink is darker. paper*0.55
    # missed the blanks and then 1a alignment collapsed.
    cut = paper * 0.68
    band = 2
    if min(h, w) < band * 2 + 4:
        return False
    sides = [gray[:band], gray[-band:], gray[:, :band], gray[:, -band:]]
    fracs = [float((side < cut).mean()) for side in sides]
    if min(fracs) < 0.40:
        return False
    inset = max(band + 1, int(round(min(h, w) * 0.28)))
    interior = gray[inset : h - inset, inset : w - inset]
    if interior.size == 0:
        return False
    inn_d = float((interior < cut).mean())
    cy, cx = h // 2, w // 2
    center = float(gray[max(0, cy - 2) : cy + 3, max(0, cx - 2) : cx + 3].mean())
    hollow = center > paper * 0.78 and inn_d <= 0.14
    slash = 0.06 <= inn_d <= 0.42
    return bool(hollow or slash)


def looks_like_printed_square(crop: np.ndarray) -> bool:
    """Four-sided ~18–24 px annulus (empty or ink-in-ring). Letter loops fail."""
    gray = _as_gray(crop)
    if gray.size == 0 or min(gray.shape[:2]) < 8:
        return False
    h, w = gray.shape[:2]
    if max(h, w) / float(min(h, w)) > 1.45:
        return False
    if _annulus_at(gray):
        return True
    # Padded 1c crops: only the centered checkbox-sized window (no slide — that
    # accepted counters inside HbA1c / Body).
    for side in (24, 22, 20, 18, 16):
        if side > h or side > w:
            continue
        y0 = max(0, (h - side) // 2)
        x0 = max(0, (w - side) // 2)
        if _annulus_at(gray[y0 : y0 + side, x0 : x0 + side]):
            return True
    return False


def refine_square_bbox(gray: np.ndarray, box: list[int], *, search: int = 10) -> list[int] | None:
    """Nudge a seed window onto a four-sided printed square, if one is nearby."""
    work = _as_gray(gray)
    h, w = work.shape[:2]
    hx, hy = int(box[0]), int(box[1])
    best: list[int] | None = None
    best_d = 10**9
    for side in (18, 20, 16, 22, 24):
        for dy in range(-search, search + 1):
            for dx in range(-search, search + 1):
                xa, ya = hx + dx, hy + dy
                if xa < 0 or ya < 0 or xa + side > w or ya + side > h:
                    continue
                if not _annulus_at(work[ya : ya + side, xa : xa + side]):
                    continue
                d = dx * dx + dy * dy
                if d < best_d:
                    best_d = d
                    best = [xa, ya, xa + side, ya + side]
    return best


def filter_checkbox_boxes(image: np.ndarray, boxes: list[list[int]]) -> list[list[int]]:
    """Drop letter-sized blobs (~30 px). Keep checkbox-sized detections (v0 ~17 px).

    Do not require a four-sided annulus here: photo detect often returns the
    bright interior of a ring, which fails a border test but is still the box.
    """
    kept: list[list[int]] = []
    for b in boxes:
        bw, bh = int(b[2] - b[0]), int(b[3] - b[1])
        if bw < CHECKBOX_SIDE_MIN or bh < CHECKBOX_SIDE_MIN:
            continue
        if bw > CHECKBOX_SIDE_MAX or bh > CHECKBOX_SIDE_MAX:
            continue
        if max(bw, bh) / float(max(1, min(bw, bh))) > 1.25:
            continue
        kept.append(b)
    return kept


def _iou(a: list[int], b: list[int]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1, area_a + area_b - inter)


def _nms(boxes: list[list[int]], iou: float = 0.4) -> list[list[int]]:
    if not boxes:
        return []
    scores = [-(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    order = sorted(range(len(boxes)), key=lambda i: scores[i])
    keep: list[list[int]] = []
    while order:
        i = order.pop()
        a = boxes[i]
        keep.append(a)
        order = [j for j in order if _iou(a, boxes[j]) < iou]
    return keep


def _snap_hollow_square_photo(
    gray: np.ndarray, box: list[int], paper: float, search: int = 6
) -> list[int]:
    h, w = gray.shape
    x0, y0, x1, y1 = (int(v) for v in box)
    side0 = max(8, min(18, (x1 - x0 + y1 - y0) // 2))
    best: list[int] | None = None
    best_score = -1e9
    for side in range(max(9, side0 - 2), min(20, side0 + 4)):
        for dy in range(-search, search + 1):
            for dx in range(-search, search + 1):
                xa, ya = x0 + dx, y0 + dy
                if xa < 0 or ya < 0 or xa + side > w or ya + side > h:
                    continue
                win = gray[ya : ya + side, xa : xa + side]
                border = np.concatenate(
                    [win[0], win[-1], win[1:-1, 0], win[1:-1, -1]]
                )
                interior = win[2:-2, 2:-2]
                if interior.size == 0:
                    continue
                local_paper = float(np.percentile(win, 90)) if win.size else paper
                int_floor = max(90.0, local_paper * 0.45)
                border_floor = max(40.0, local_paper * 0.25)
                if float(interior.min()) < int_floor or float(border.min()) < border_floor:
                    continue
                score = float(interior.mean()) - float(border.mean())
                if score < 4:
                    continue
                if score > best_score:
                    best_score = score
                    best = [xa, ya, xa + side, ya + side]
    return best if best is not None else [x0, y0, x1, y1]


def detect_checkboxes_photo(image: Image.Image | np.ndarray) -> list[list[int]]:
    """Find hollow gray checkbox rings on a photographed sheet."""
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            gray = arr
        else:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)

    paper = float(np.percentile(gray, 92))
    h, w = gray.shape
    y_lo = int(0.08 * h)
    scale = 2.0
    work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    work = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(work)
    blur = cv2.GaussianBlur(work, (3, 3), 0)
    edges = cv2.dilate(
        cv2.Canny(blur, 15, 60),
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    hh, ww = work.shape
    min_side = max(14, int(min(hh, ww) * 0.007))
    max_side = max(32, int(min(hh, ww) * 0.028))
    boxes: list[list[int]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < min_side or bh < min_side or bw > max_side or bh > max_side:
            continue
        ratio = bw / max(bh, 1)
        if ratio < 0.7 or ratio > 1.35:
            continue
        if y < int(y_lo * scale) or y > int(hh * 0.96):
            continue
        interior = work[y + 2 : y + bh - 2, x + 2 : x + bw - 2]
        if interior.size == 0:
            continue
        border = np.concatenate(
            [
                work[y, x : x + bw],
                work[y + bh - 1, x : x + bw],
                work[y : y + bh, x],
                work[y : y + bh, x + bw - 1],
            ]
        )
        local_paper = float(np.percentile(work[y : y + bh, x : x + bw], 90))
        int_min = local_paper * 0.55
        int_mean = local_paper * 0.76
        border_min = local_paper * 0.32
        if float(interior.min()) < int_min or float(interior.mean()) < int_mean:
            continue
        if float(border.min()) < border_min:
            continue
        if float(interior.mean()) - float(border.mean()) < 4:
            continue
        boxes.append(
            [
                int(round(x / scale)),
                int(round(y / scale)),
                int(round((x + bw) / scale)),
                int(round((y + bh) / scale)),
            ]
        )
    boxes = _nms(boxes, iou=0.4)
    boxes = [_snap_hollow_square_photo(gray, b, paper) for b in boxes]
    boxes = [b for b in boxes if b[1] >= y_lo]
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes

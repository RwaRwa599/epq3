"""Page quad detection, perspective warp, and orientation correction."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from med_doc.paths import CANONICAL_SIZE


def load_rgb(image: Image.Image | np.ndarray | str) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        return arr
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = ImageOps.exif_transpose(image.convert("RGB"))
    return np.asarray(image)


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    s = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    total = s.sum(axis=1)
    diff = np.diff(s, axis=1).ravel()
    tl = s[int(np.argmin(total))]
    br = s[int(np.argmax(total))]
    tr = s[int(np.argmin(diff))]
    bl = s[int(np.argmax(diff))]
    return np.stack([tl, tr, br, bl]).astype(np.float32)


def _quad_size(quad: np.ndarray) -> tuple[float, float]:
    tl, tr, br, bl = order_quad(quad)
    width = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))
    height = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    return float(width), float(height)


def _rotate_quad(quad: np.ndarray, k: int) -> np.ndarray:
    """Rotate ordered TL,TR,BR,BL by k*90 degrees clockwise."""
    ordered = order_quad(quad)
    k = k % 4
    if k == 0:
        return ordered
    if k == 1:  # 90 CW: TL<-BL, TR<-TL, BR<-TR, BL<-BR
        return np.stack([ordered[3], ordered[0], ordered[1], ordered[2]])
    if k == 2:
        return np.stack([ordered[2], ordered[3], ordered[0], ordered[1]])
    return np.stack([ordered[1], ordered[2], ordered[3], ordered[0]])


def detect_document_quad(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Find the page quadrilateral in RGB/BGR-agnostic RGB array (RGB)."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    scale = min(1.0, 800.0 / max(h, w))
    work = gray
    if scale != 1.0:
        work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    blur = cv2.GaussianBlur(work, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = float(work[otsu == 255].mean()) if np.any(otsu == 255) else 0.0
    dark = float(work[otsu == 0].mean()) if np.any(otsu == 0) else 255.0
    if bright < dark:
        otsu = 255 - otsu
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    meta: dict[str, Any] = {"method": "full-frame", "confidence": 0.35}
    full = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    work_area = float(work.shape[0] * work.shape[1])
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if 0.10 * work_area <= area <= 0.99 * work_area:
            candidates.append((area, contour))
    if not candidates:
        return full, meta

    page = max(candidates, key=lambda item: item[0])[1]
    area = float(cv2.contourArea(page))
    peri = cv2.arcLength(page, True)
    approx = cv2.approxPolyDP(page, 0.02 * peri, True)
    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float32) / scale
        if _is_full_frame(quad, w, h):
            return full, meta
        meta = {
            "method": "page-quad",
            "confidence": float(np.clip(area / work_area, 0.0, 1.0)),
        }
        return order_quad(quad), meta

    rect = cv2.minAreaRect(page)
    box = cv2.boxPoints(rect).astype(np.float32) / scale
    if _is_full_frame(box, w, h):
        return full, meta
    meta = {
        "method": "min-area-rect",
        "confidence": float(np.clip(min(0.85, area / work_area), 0.0, 1.0)),
    }
    return order_quad(box), meta


def _is_full_frame(quad: np.ndarray, w: int, h: int, margin: float = 0.02) -> bool:
    pts = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    mx, my = margin * w, margin * h
    near_x = np.mean((pts[:, 0] < mx) | (pts[:, 0] > w - 1 - mx))
    near_y = np.mean((pts[:, 1] < my) | (pts[:, 1] > h - 1 - my))
    return bool(near_x > 0.7 and near_y > 0.7)


def letterbox_to_canvas(
    image: np.ndarray,
    dest_size: tuple[int, int],
) -> np.ndarray:
    """Uniform-scale by height, then pad or crop width — never stretch Y.

    Full-page scans that are already ~canonical width but a few dozen pixels
    short/tall must not be anisotropically resized: that pitches the overlay
    onto printed labels.
    """
    tw, th = dest_size
    if image.shape[1] == tw and image.shape[0] == th:
        return image
    h, w = image.shape[:2]
    if h < 1 or w < 1:
        return np.full((th, tw, 3), 255, dtype=np.uint8)
    scale = th / h
    nw = max(1, int(round(w * scale)))
    resized = cv2.resize(image, (nw, th), interpolation=cv2.INTER_CUBIC)
    if nw == tw:
        return resized
    if nw < tw:
        canvas = np.full((th, tw, 3), 255, dtype=np.uint8)
        x0 = (tw - nw) // 2
        canvas[:, x0 : x0 + nw] = resized
        return canvas
    x0 = (nw - tw) // 2
    return resized[:, x0 : x0 + tw]


def _quad_is_full_page_scan(quad: np.ndarray, w: int, h: int) -> bool:
    xs, ys = quad[:, 0], quad[:, 1]
    return bool((xs.max() - xs.min()) / max(w, 1) > 0.90 and (ys.max() - ys.min()) / max(h, 1) > 0.82)


def four_point_transform(
    image: np.ndarray,
    quad: np.ndarray,
    dest_size: tuple[int, int] = CANONICAL_SIZE,
) -> np.ndarray:
    tw, th = dest_size
    src = order_quad(quad)
    dst = np.array(
        [[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image, matrix, (tw, th), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
    )


def _header_score(gray: np.ndarray) -> float:
    """Prefer orientations with a printed header (horizontal structure) at the top."""
    h, w = gray.shape
    top = gray[: max(1, int(h * 0.10))]
    bottom = gray[int(h * 0.90) :]
    # Horizontal edges in the header band, darkness vs the footer.
    top_edges = cv2.Sobel(top, cv2.CV_32F, 0, 1, ksize=3)
    bot_edges = cv2.Sobel(bottom, cv2.CV_32F, 0, 1, ksize=3)
    edge_ratio = float(np.mean(np.abs(top_edges))) / (float(np.mean(np.abs(bot_edges))) + 1e-6)
    dark_ratio = (255.0 - float(top.mean())) / (255.0 - float(bottom.mean()) + 1e-6)
    # Four checkbox columns → peaks in a vertical projection of the upper body.
    body = gray[int(h * 0.10) : int(h * 0.85)]
    ink = (body < 200).astype(np.float32)
    proj = ink.mean(axis=0)
    kernel = max(3, w // 80)
    smooth = np.convolve(proj, np.ones(kernel) / kernel, mode="same")
    peaks = 0
    for i in range(1, len(smooth) - 1):
        if smooth[i] > 0.04 and smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]:
            peaks += 1
    peak_score = 1.0 - abs(min(peaks, 8) - 4) / 4.0
    return float(0.4 * min(edge_ratio, 3.0) / 3.0 + 0.3 * min(dark_ratio, 3.0) / 3.0 + 0.3 * peak_score)


def correct_orientation(
    canvas: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Pick 0° or 180° so the printed header sits at the top.

    90/270 are handled by rotating the source quad before the warp. Rotating
    an already-canonical landscape canvas and resizing it back would squash
    the form and destroy ROI alignment.
    """
    best_img = canvas
    best_deg = 0
    best_score = -1e9
    for deg in (0, 180):
        rot = canvas if deg == 0 else cv2.rotate(canvas, cv2.ROTATE_180)
        g = cv2.cvtColor(rot, cv2.COLOR_RGB2GRAY) if rot.ndim == 3 else rot
        score = _header_score(g)
        if deg == 0:
            score += 0.02  # stable tie-break: keep the warp assignment
        if score > best_score:
            best_score = score
            best_img = rot
            best_deg = deg
    return best_img, best_deg, float(np.clip(best_score, 0.0, 1.0))


def align_quad_to_canvas_aspect(
    quad: np.ndarray,
    dest_size: tuple[int, int] = CANONICAL_SIZE,
) -> np.ndarray:
    """If the detected page is portrait relative to the landscape form, rotate the quad."""
    width, height = _quad_size(quad)
    dest_aspect = dest_size[0] / max(dest_size[1], 1)
    page_aspect = width / max(height, 1.0)
    if page_aspect < 1.0 and dest_aspect > 1.0:
        return _rotate_quad(quad, 1)
    if page_aspect > 1.0 and dest_aspect < 1.0:
        return _rotate_quad(quad, 1)
    return order_quad(quad)


def warp_to_canonical(
    image: Image.Image | np.ndarray | str,
    dest_size: tuple[int, int] = CANONICAL_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    rgb = load_rgb(image)
    quad, detect_meta = detect_document_quad(rgb)
    quad = align_quad_to_canvas_aspect(quad, dest_size)
    warped = four_point_transform(rgb, quad, dest_size)
    oriented, degrees, orient_score = correct_orientation(warped)
    meta = {
        **detect_meta,
        "orientation_degrees": degrees,
        "orientation_score": orient_score,
        "quad": quad.tolist(),
    }
    meta["confidence"] = float(
        0.6 * float(detect_meta.get("confidence", 0.5)) + 0.4 * orient_score
    )
    return oriented, meta

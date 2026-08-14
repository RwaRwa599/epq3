"""Adaptive field crops and illumination-invariant normalization."""

from __future__ import annotations

import cv2
import numpy as np

from med_doc.schemas import FieldCrop, FieldSpec, TemplateSpec


def quality_metrics(gray: np.ndarray) -> tuple[float, float, float]:
    """Return (quality, blur_score, glare_index) in [0, 1]."""
    if gray.size == 0:
        return 0.0, 0.0, 1.0
    work = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(work, cv2.CV_64F).var())
    blur_score = float(np.clip(lap_var / 250.0, 0.0, 1.0))
    glare_index = float(np.mean(work >= 250))
    quality = float(np.clip(blur_score * (1.0 - glare_index), 0.0, 1.0))
    return quality, blur_score, glare_index


def clahe_normalize(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def background_divide(gray: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Local background division (illumination flattening)."""
    k = max(3, int(sigma) * 2 + 1)
    background = cv2.GaussianBlur(gray, (k, k), sigma)
    background = np.maximum(background, 1)
    divided = cv2.divide(gray, background, scale=255)
    return divided


def normalize_crop_rgb(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    flat = background_divide(gray)
    return clahe_normalize(flat)


def _apply_column_shift(
    bbox: list[int],
    spec: FieldSpec,
    template: TemplateSpec,
    column_shifts: dict[int, float],
) -> list[int]:
    if not column_shifts:
        return bbox
    boxes = template.checkbox_fields()
    xs = [0.5 * (f.bbox[0] + f.bbox[2]) for f in boxes]
    if not xs:
        return bbox
    cuts = np.quantile(xs, [0.25, 0.5, 0.75])
    cx = 0.5 * (spec.bbox[0] + spec.bbox[2])
    col = int(np.searchsorted(cuts, cx))
    dy = int(round(column_shifts.get(col, 0.0)))
    if dy == 0:
        return bbox
    h = template.height
    return [bbox[0], max(0, bbox[1] + dy), bbox[2], min(h, bbox[3] + dy)]


def clip_bbox(bbox: list[int], w: int, h: int) -> list[int]:
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    return [x0, y0, x1, y1]


def crop_field(
    canvas: np.ndarray,
    spec: FieldSpec,
    template: TemplateSpec,
    column_shifts: dict[int, float] | None = None,
) -> FieldCrop:
    h, w = canvas.shape[:2]
    bbox = clip_bbox(template.pixel_bbox(spec, apply_pad=True), w, h)
    if spec.field_type == "checkbox" and column_shifts:
        bbox = clip_bbox(_apply_column_shift(bbox, spec, template, column_shifts), w, h)
    raw = canvas[bbox[1] : bbox[3], bbox[0] : bbox[2]].copy()
    if raw.size == 0:
        raw = np.full((4, 4, 3), 255, dtype=np.uint8)
    normalized = normalize_crop_rgb(raw)
    quality, blur, glare = quality_metrics(normalized)
    return FieldCrop(
        field_id=spec.field_id,
        field_type=spec.field_type,
        canonical_bbox=bbox,
        normalized_image=normalized,
        raw_image=raw,
        quality_score=quality,
        blur_score=blur,
        glare_index=glare,
    )


def extract_crops(
    canvas: np.ndarray,
    template: TemplateSpec,
    column_shifts: dict[int, float] | None = None,
) -> tuple[dict[str, FieldCrop], dict[str, FieldCrop]]:
    checkbox: dict[str, FieldCrop] = {}
    handwriting: dict[str, FieldCrop] = {}
    for spec in template.fields:
        crop = crop_field(canvas, spec, template, column_shifts)
        if spec.field_type == "checkbox":
            checkbox[spec.field_id] = crop
        else:
            handwriting[spec.field_id] = crop
    return checkbox, handwriting

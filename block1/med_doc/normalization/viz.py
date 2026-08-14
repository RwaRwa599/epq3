"""Debug overlays for Block 1 crops and landmarks."""

from __future__ import annotations

import cv2
import numpy as np

from med_doc.schemas import NormalizedDocumentResult, TemplateSpec


def draw_overlay(
    canvas: np.ndarray,
    result: NormalizedDocumentResult,
    template: TemplateSpec | None = None,
) -> np.ndarray:
    vis = canvas.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)

    def _box(bbox: list[int], color: tuple[int, int, int], width: int = 1) -> None:
        x0, y0, x1, y1 = (int(v) for v in bbox)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, width)

    for crop in result.checkbox_crops.values():
        _box(crop.canonical_bbox, (40, 180, 80), 1)
    for crop in result.handwriting_crops.values():
        _box(crop.canonical_bbox, (40, 90, 220), 2)
    if template is not None:
        h, w = vis.shape[:2]
        for landmark in template.landmarks:
            x0 = int(landmark.bbox[0] * w)
            y0 = int(landmark.bbox[1] * h)
            x1 = int(landmark.bbox[2] * w)
            y1 = int(landmark.bbox[3] * h)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (220, 160, 40), 2)
    return vis

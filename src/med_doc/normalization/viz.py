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

    if template is not None:
        vis = draw_sections(vis, template)
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
            cv2.rectangle(vis, (x0, y0), (x1, y1), (220, 160, 40), 1)
    return vis


def draw_sections(canvas: np.ndarray, template: TemplateSpec) -> np.ndarray:
    """Main sections (magenta) and bold-subhead territories (cyan)."""
    vis = canvas.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    h, w = vis.shape[:2]

    def _rel_box(bbox: list[float]) -> tuple[int, int, int, int]:
        return (
            int(round(bbox[0] * w)),
            int(round(bbox[1] * h)),
            int(round(bbox[2] * w)),
            int(round(bbox[3] * h)),
        )

    for spec in template.main_sections():
        x0, y0, x1, y1 = _rel_box(spec.bbox)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (200, 40, 180), 3)
        cv2.putText(
            vis,
            spec.label,
            (x0 + 4, max(14, y0 + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 40, 180),
            1,
            cv2.LINE_AA,
        )
    for spec in template.sub_sections():
        x0, y0, x1, y1 = _rel_box(spec.bbox)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (20, 160, 200), 2)
        cv2.putText(
            vis,
            spec.label,
            (x0 + 4, min(h - 4, y0 + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (20, 120, 180),
            1,
            cv2.LINE_AA,
        )
    for spec in template.row_sections():
        x0, y0, x1, y1 = _rel_box(spec.bbox)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (40, 190, 80), 1)
    for spec in template.text_sections():
        x0, y0, x1, y1 = _rel_box(spec.bbox)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (220, 140, 30), 1)
    for spec in template.tick_sections():
        x0, y0, x1, y1 = _rel_box(spec.bbox)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (220, 40, 40), 2)
    return vis

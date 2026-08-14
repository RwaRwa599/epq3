"""Color-coded Block 3 overlay: green = high confidence, amber = HiTL, gray = unmarked."""

from __future__ import annotations

import cv2
import numpy as np

from med_doc.htr.schemas import DocumentPrediction

GREEN = (40, 180, 80)
AMBER = (220, 160, 40)
GRAY = (140, 140, 140)
BLUE = (40, 90, 220)
RED = (200, 50, 50)


def _box(vis: np.ndarray, bbox: list[int], color: tuple[int, int, int], width: int = 2) -> None:
    x0, y0, x1, y1 = (int(v) for v in bbox)
    cv2.rectangle(vis, (x0, y0), (x1, y1), color, width)


def _label(vis: np.ndarray, bbox: list[int], text: str, color: tuple[int, int, int]) -> None:
    x0, y0, _, _ = (int(v) for v in bbox)
    caption = (text or "")[:28]
    if not caption:
        return
    y = max(12, y0 - 4)
    cv2.putText(vis, caption, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def draw_prediction_overlay(
    canvas: np.ndarray,
    prediction: DocumentPrediction,
    fields_meta: dict,
) -> np.ndarray:
    """Draw mark and handwriting predictions onto the canonical canvas."""
    vis = canvas.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)

    checkboxes = (fields_meta or {}).get("checkboxes", {})
    handwriting = (fields_meta or {}).get("handwriting", {})

    for fid, mark in prediction.checkbox_marks.items():
        meta = checkboxes.get(fid, {})
        bbox = meta.get("bbox")
        if not bbox:
            continue
        if mark.needs_hitl:
            color = AMBER
        elif mark.is_marked:
            color = GREEN
        else:
            color = GRAY
        _box(vis, bbox, color, 1)

    for fid, hw in prediction.handwriting_fields.items():
        meta = handwriting.get(fid, {})
        bbox = meta.get("bbox")
        if not bbox:
            continue
        color = AMBER if hw.needs_hitl else (BLUE if (hw.canonical_value or hw.raw_text) else GRAY)
        _box(vis, bbox, color, 2)
        shown = hw.canonical_value or hw.raw_text or fid
        _label(vis, bbox, shown, color)

    return vis

"""Block 1b: section layout — lock, per-section dy, squares, extra-ink, text crops."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from med_doc.normalization.align import match_landmark, snap_overlay
from med_doc.normalization.block1a import Block1aPage
from med_doc.normalization.crops import (
    apply_column_shifts_to_template,
    extract_crops,
    extract_section_crops,
)
from med_doc.normalization.gates import MIN_ALIGNMENT_CONFIDENCE
from med_doc.normalization.sections import apply_section_dy, apply_tick_windows, snap_sections
from med_doc.normalization.viz import draw_overlay
from med_doc.schemas import FieldSpec, NormalizedDocumentResult, TemplateSpec


@dataclass
class Block1bLayout:
    result: NormalizedDocumentResult
    sectioned: TemplateSpec

    def to_result(self) -> NormalizedDocumentResult:
        return self.result


def _shift_hw_y(spec: FieldSpec, dy_rel: float) -> FieldSpec:
    x0, y0, x1, y1 = spec.bbox
    h = y1 - y0
    y0 = min(max(0.0, y0 + dy_rel), 0.995)
    y1 = min(1.0, max(y0 + 0.004, y0 + h))
    return spec.model_copy(update={"bbox": [x0, round(y0, 6), x1, round(y1, 6)]})


def snap_handwriting_fields(canvas: np.ndarray, template: TemplateSpec) -> tuple[TemplateSpec, dict]:
    """Nudge tube/office/header write-ins using footer/header landmarks.

    Checkbox snap does not move handwriting; overlay Y for tubes sat on the last
    cardiovascular labels (NT-proBNP / Lipoprotein (a)).
    """
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY) if canvas.ndim == 3 else canvas
    h = float(template.height)
    meta: dict = {"footer_dy": 0.0, "header_dy": 0.0, "n_shifted": 0}
    footer_dy = 0.0
    header_dy = 0.0
    for landmark in template.landmarks:
        if landmark.kind == "footer_rule":
            _dx, dy, ncc = match_landmark(gray, landmark.bbox, search_frac=0.10, mode="footer_rule")
            if ncc >= 0.15:
                footer_dy = float(dy)
                meta["footer_ncc"] = round(float(ncc), 3)
        elif landmark.kind == "header_bar":
            _dx, dy, ncc = match_landmark(gray, landmark.bbox, search_frac=0.06, mode="header_bar")
            if ncc >= 0.15:
                header_dy = float(dy)
                meta["header_ncc"] = round(float(ncc), 3)
    meta["footer_dy"] = round(footer_dy, 2)
    meta["header_dy"] = round(header_dy, 2)
    if abs(footer_dy) < 2 and abs(header_dy) < 2:
        return template, meta
    fields: list[FieldSpec] = []
    n = 0
    for spec in template.fields:
        if spec.field_type == "checkbox":
            fields.append(spec)
            continue
        dy = 0.0
        group = spec.group or ""
        if group in {"tube", "office"} or spec.field_id.startswith("tube_"):
            dy = footer_dy
        elif spec.field_id == "clinical_info":
            dy = header_dy
        if abs(dy) >= 2:
            fields.append(_shift_hw_y(spec, dy / h))
            n += 1
        else:
            fields.append(spec)
    meta["n_shifted"] = n
    return template.model_copy(update={"fields": fields}), meta


def run_block1b(page: Block1aPage, *, draw_debug: bool = True) -> Block1bLayout:
    """Section lock + squares + extra-ink. Does not classify ticks."""
    aligned = page.canvas
    spec = page.template
    col_shifts = page.col_shifts

    snapped, snap_meta = snap_overlay(aligned, spec)
    snapped = apply_column_shifts_to_template(snapped, col_shifts)
    snapped, hw_meta = snap_handwriting_fields(aligned, snapped)
    sectioned, section_meta = snap_sections(aligned, snapped)
    shifted, dy_meta = apply_section_dy(aligned, sectioned)
    gated, crop_meta = apply_tick_windows(aligned, shifted)
    snap_grid = float(snap_meta.get("grid_score", 0) or 0)
    # Gate only square placement. Extra-ink / overlay always keep section lock + dy.
    if crop_meta["grid"] + 1e-9 >= snap_grid - 0.02:
        crop_src = gated
        crop_meta["used"] = True
    else:
        crop_src = shifted
        crop_meta["used"] = False
    crop_meta["snap_grid"] = round(snap_grid, 4)
    crop_meta["kept_section_lock"] = True

    checkbox, _ = extract_crops(aligned, crop_src, column_shifts={})
    # Section dy only moves checkboxes. Write-ins come from the footer/header snap.
    _, handwriting = extract_crops(aligned, snapped, column_shifts={})
    section_crops = extract_section_crops(aligned, shifted)

    confidence = float(
        np.clip(
            0.4 * float(page.warp_meta.get("confidence", 0.5))
            + 0.3 * float(page.align_meta.get("confidence", 0.5))
            + 0.3 * float(snap_meta.get("confidence", snap_meta.get("grid_score", 0.5))),
            0.0,
            1.0,
        )
    )
    result = NormalizedDocumentResult(
        document_id=page.document_id,
        canonical_canvas=aligned,
        alignment_confidence=confidence,
        checkbox_crops=checkbox,
        handwriting_crops=handwriting,
        section_crops=section_crops,
        warp_method=page.warp_method,
        orientation_degrees=page.orientation_degrees,
        extra={
            "warp": page.warp_meta,
            "align": page.align_meta,
            "snap": snap_meta,
            "handwriting_snap": hw_meta,
            "section_snap": section_meta,
            "section_dy": dy_meta,
            "tick_crops": crop_meta,
            "registration": {
                "grid_score": crop_meta.get("grid"),
                "snap_grid": snap_grid,
                "hollow_hit": crop_meta.get("grid"),
                "section_dy": dy_meta.get("shifts", {}),
            },
            "template_id": spec.template_id,
            "template_pick": page.extra.get("template_pick") or {},
            "n_checkbox": len(spec.checkbox_fields()),
            "page_gate": page.extra.get("page_gate") or {},
            "needs_review": bool(page.extra.get("needs_review"))
            or confidence < MIN_ALIGNMENT_CONFIDENCE,
            "sections": [s.model_dump() for s in shifted.sections],
        },
    )
    if draw_debug:
        result.debug_overlay = draw_overlay(aligned, result, shifted)
    return Block1bLayout(result=result, sectioned=shifted)

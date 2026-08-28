"""Block 1b: section layout — lock, per-section dy, squares, extra-ink, text crops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from med_doc.normalization.align import snap_overlay
from med_doc.normalization.block1a import Block1aPage
from med_doc.normalization.crops import (
    apply_column_shifts_to_template,
    extract_crops,
    extract_section_crops,
)
from med_doc.normalization.sections import apply_section_dy, apply_tick_windows, snap_sections
from med_doc.normalization.viz import draw_overlay
from med_doc.schemas import NormalizedDocumentResult, TemplateSpec


@dataclass
class Block1bLayout:
    result: NormalizedDocumentResult
    sectioned: TemplateSpec

    def to_result(self) -> NormalizedDocumentResult:
        return self.result


def run_block1b(page: Block1aPage, *, draw_debug: bool = True) -> Block1bLayout:
    """Section lock + squares + extra-ink. Does not classify ticks."""
    aligned = page.canvas
    spec = page.template
    col_shifts = page.col_shifts

    snapped, snap_meta = snap_overlay(aligned, spec)
    snapped = apply_column_shifts_to_template(snapped, col_shifts)
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
            "sections": [s.model_dump() for s in shifted.sections],
        },
    )
    if draw_debug:
        result.debug_overlay = draw_overlay(aligned, result, shifted)
    return Block1bLayout(result=result, sectioned=shifted)

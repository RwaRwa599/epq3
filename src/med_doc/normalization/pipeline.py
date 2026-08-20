"""Block 1 public API: raw photo → canonical canvas + field crops."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from med_doc.normalization.align import _checkbox_grid_score, fine_align, snap_overlay
from med_doc.normalization.crops import extract_crops
from med_doc.normalization.sections import apply_tick_windows, snap_sections
from med_doc.normalization.viz import draw_overlay
from med_doc.normalization.warp import warp_to_canonical
from med_doc.paths import DEFAULT_TEMPLATE, V1_TEMPLATE
from med_doc.schemas import NormalizedDocumentResult, TemplateSpec
from med_doc.template import load_template


def _score_overlay(canvas: np.ndarray, spec: TemplateSpec) -> float:
    gray = canvas if canvas.ndim == 2 else canvas.mean(axis=2)
    return _checkbox_grid_score(gray.astype(np.uint8), spec)


def _pick_revision(
    image: Image.Image | np.ndarray | str | Path,
    explicit: TemplateSpec | str | Path | None,
) -> TemplateSpec:
    if isinstance(explicit, TemplateSpec):
        return explicit
    if explicit is not None:
        return load_template(explicit)
    v0 = load_template(DEFAULT_TEMPLATE)
    if not V1_TEMPLATE.exists():
        return v0
    v1 = load_template(V1_TEMPLATE)
    canvas0, _ = warp_to_canonical(image, dest_size=(v0.width, v0.height))
    canvas1, _ = warp_to_canonical(image, dest_size=(v1.width, v1.height))
    _, s0_meta = snap_overlay(canvas0, v0)
    _, s1_meta = snap_overlay(canvas1, v1)
    score0 = s0_meta.get("n_snapped", 0) + 0.5 * s0_meta.get("n_offset_hits", 0)
    score1 = s1_meta.get("n_snapped", 0) + 0.5 * s1_meta.get("n_offset_hits", 0)
    return v1 if score1 >= score0 else v0


def normalize_document(
    image: Image.Image | np.ndarray | str | Path,
    template: TemplateSpec | str | Path | None = None,
    document_id: str | None = None,
    draw_debug: bool = True,
) -> NormalizedDocumentResult:
    if document_id is None:
        if isinstance(image, (str, Path)):
            document_id = Path(image).stem
        else:
            document_id = "page"

    spec = _pick_revision(image, template)
    dest = (spec.width, spec.height)
    canvas, warp_meta = warp_to_canonical(image, dest_size=dest)
    aligned, align_meta, col_shifts = fine_align(canvas, spec)
    snapped, snap_meta = snap_overlay(aligned, spec)
    sectioned, section_meta = snap_sections(aligned, snapped)
    gated, crop_meta = apply_tick_windows(aligned, sectioned)
    snap_grid = float(snap_meta.get("grid_score", 0) or 0)
    if crop_meta["grid"] + 1e-9 >= snap_grid - 0.02:
        crop_src = gated
        crop_meta["used"] = True
    else:
        crop_src = snapped
        crop_meta["used"] = False
    checkbox, handwriting = extract_crops(aligned, crop_src, col_shifts)

    confidence = float(
        np.clip(
            0.4 * float(warp_meta.get("confidence", 0.5))
            + 0.3 * float(align_meta.get("confidence", 0.5))
            + 0.3 * float(snap_meta.get("confidence", align_meta.get("grid_score", 0.5))),
            0.0,
            1.0,
        )
    )
    result = NormalizedDocumentResult(
        document_id=document_id,
        canonical_canvas=aligned,
        alignment_confidence=confidence,
        checkbox_crops=checkbox,
        handwriting_crops=handwriting,
        warp_method=str(warp_meta.get("method", "none")),
        orientation_degrees=int(warp_meta.get("orientation_degrees", 0)),
        extra={
            "warp": warp_meta,
            "align": align_meta,
            "snap": snap_meta,
            "section_snap": section_meta,
            "tick_crops": crop_meta,
            "template_id": spec.template_id,
            "sections": [s.model_dump() for s in sectioned.sections],
        },
    )
    if draw_debug:
        result.debug_overlay = draw_overlay(aligned, result, sectioned)
    return result

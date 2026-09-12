"""Block 1a: global page normalisation (revision pick, warp, page-level fine_align)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from med_doc.normalization.align import fine_align, snap_overlay
from med_doc.normalization.warp import warp_to_canonical
from med_doc.paths import DEFAULT_TEMPLATE, V1_TEMPLATE
from med_doc.schemas import TemplateSpec
from med_doc.template import load_template


@dataclass
class Block1aPage:
    """Canonical canvas after warp + page-level fine alignment."""

    canvas: np.ndarray
    template: TemplateSpec
    warp_meta: dict
    align_meta: dict
    col_shifts: dict[int, float]
    document_id: str
    warp_method: str = "none"
    orientation_degrees: int = 0
    extra: dict = field(default_factory=dict)


def pick_revision(
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


def run_block1a(
    image: Image.Image | np.ndarray | str | Path,
    template: TemplateSpec | str | Path | None = None,
    document_id: str | None = None,
) -> Block1aPage:
    """Pick v0/v1, warp to canonical, page-level fine_align. No section layout."""
    if document_id is None:
        if isinstance(image, (str, Path)):
            document_id = Path(image).stem
        else:
            document_id = "page"

    spec = pick_revision(image, template)
    dest = (spec.width, spec.height)
    canvas, warp_meta = warp_to_canonical(image, dest_size=dest)
    aligned, align_meta, col_shifts = fine_align(canvas, spec)
    return Block1aPage(
        canvas=aligned,
        template=spec,
        warp_meta=warp_meta,
        align_meta=align_meta,
        col_shifts=col_shifts,
        document_id=document_id,
        warp_method=str(warp_meta.get("method", "none")),
        orientation_degrees=int(warp_meta.get("orientation_degrees", 0)),
        extra={"warp": warp_meta, "align": align_meta},
    )

"""Block 1 public API: raw photo → canonical canvas + field crops."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from med_doc.normalization.block1a import pick_revision, run_block1a
from med_doc.normalization.block1b import run_block1b
from med_doc.normalization.block1c import run_block1c
from med_doc.schemas import NormalizedDocumentResult, TemplateSpec

# Back-compat alias used by continuity / older call sites.
_pick_revision = pick_revision


def normalize_document(
    image: Image.Image | np.ndarray | str | Path,
    template: TemplateSpec | str | Path | None = None,
    document_id: str | None = None,
    draw_debug: bool = True,
) -> NormalizedDocumentResult:
    """1a warp → 1b layout → 1c crop gate. Does not classify ticks."""
    page = run_block1a(image, template, document_id=document_id)
    layout = run_block1b(page, draw_debug=draw_debug)
    gated = run_block1c(page, layout, draw_debug=draw_debug)
    return gated.to_result()

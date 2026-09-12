"""Block 1a: global page normalisation (revision pick, warp, page-level fine_align)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from med_doc.normalization.align import fine_align, snap_overlay
from med_doc.normalization.inputs import collect_image_inputs
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


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def save_block1a_page(page: Block1aPage, doc_dir: str | Path) -> dict:
    """Write canonical canvas + 1a metadata (no 1b/1c crops)."""
    doc_path = Path(doc_dir)
    doc_path.mkdir(parents=True, exist_ok=True)
    rgb = page.canvas
    if rgb.ndim == 3:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        bgr = rgb
    cv2.imwrite(str(doc_path / "canonical.png"), bgr)
    meta = {
        "doc_id": page.document_id,
        "block": "block1a",
        "template_id": page.template.template_id,
        "canvas_size": [int(page.canvas.shape[1]), int(page.canvas.shape[0])],
        "warp_method": page.warp_method,
        "orientation_degrees": page.orientation_degrees,
        "col_shifts": {str(k): float(v) for k, v in page.col_shifts.items()},
        "warp": _jsonable(page.warp_meta),
        "align": _jsonable(page.align_meta),
        "canonical_path": "canonical.png",
    }
    (doc_path / "block1a.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def run_block1a_batch(
    inputs: list | str | Path,
    *,
    output_dir: str | Path | None = None,
    output_zip: str | Path | None = None,
    template: TemplateSpec | str | Path | None = None,
) -> dict:
    """Warp a folder, ZIP, or list of images through Block 1a only.

    Accepts a directory of PNG/JPEG/etc, a ZIP of those files, one file, or a
    sequence of paths / arrays. Does not run 1b layout or 1c crop gating.
    """
    import tempfile
    import zipfile

    items = collect_image_inputs(inputs)
    if output_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix="block1a_out_"))
    else:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    documents: list[dict] = []
    pages: dict[str, Block1aPage] = {}
    print(f"[Block 1a] Starting batch warp for {len(items)} document(s)...")
    for doc_id, item in items:
        try:
            print(f"  → 1a '{doc_id}'...")
            page = run_block1a(item, template=template, document_id=doc_id)
            meta = save_block1a_page(page, target_dir / "docs" / doc_id)
            pages[doc_id] = page
            documents.append({"status": "success", **meta})
        except Exception as exc:
            print(f"  [!] Error processing '{doc_id}': {exc}")
            documents.append({"doc_id": doc_id, "status": "error", "error": str(exc)})

    manifest = {
        "version": "1.0",
        "block": "block1a",
        "stage": "warp_align",
        "total_documents": len(items),
        "successful_documents": sum(1 for d in documents if d.get("status") == "success"),
        "documents": documents,
    }
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path = None
    if output_zip is not None:
        zip_path = Path(output_zip)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in target_dir.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, arcname=file_path.relative_to(target_dir))
        print(f"✓ Block 1a ZIP created: {zip_path}")

    return {
        "manifest": manifest,
        "output_dir": str(target_dir),
        "output_zip": str(zip_path) if zip_path else None,
        "pages": pages,
    }

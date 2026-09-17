"""if1block1: bar+gutter warp, then existing 1b overlay and 1c square gate."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np

from med_doc.if1.layout_warp import layout_warp
from med_doc.normalization.align import fine_align
from med_doc.normalization.batch import save_normalized_document
from med_doc.normalization.block1a import (
    Block1aPage,
    pick_revision_with_meta,
)
from med_doc.normalization.block1b import run_block1b
from med_doc.normalization.block1c import run_block1c
from med_doc.normalization.gates import alignment_gate, batch_template_conflict, template_pick_needs_review
from med_doc.normalization.inputs import collect_image_inputs
from med_doc.schemas import NormalizedDocumentResult, TemplateSpec


def normalize_document(
    image: Image.Image | np.ndarray | str | Path,
    template: TemplateSpec | str | Path | None = None,
    document_id: str | None = None,
    draw_debug: bool = True,
) -> NormalizedDocumentResult:
    """Layout warp → fine_align → 1b → 1c. Does not classify ticks."""
    if document_id is None:
        if isinstance(image, (str, Path)):
            document_id = Path(image).stem
        else:
            document_id = "page"

    spec, pick_meta = pick_revision_with_meta(image, template)
    canvas, warp_meta = layout_warp(image, spec)
    aligned, align_meta, col_shifts = fine_align(canvas, spec)
    page_gate = alignment_gate(float(align_meta.get("confidence") or 0.0))
    extra = {
        "warp": warp_meta,
        "align": align_meta,
        "template_pick": pick_meta,
        "page_gate": page_gate,
        "needs_review": (not bool(page_gate["ok"])) or template_pick_needs_review(pick_meta),
        "n_checkbox": len(spec.checkbox_fields()),
        "template_id": spec.template_id,
        "if1": True,
    }
    page = Block1aPage(
        canvas=aligned,
        template=spec,
        warp_meta=warp_meta,
        align_meta=align_meta,
        col_shifts=col_shifts,
        document_id=document_id,
        warp_method=str(warp_meta.get("method", "none")),
        orientation_degrees=int(warp_meta.get("orientation_degrees", 0)),
        extra=extra,
    )
    layout = run_block1b(page, draw_debug=draw_debug)
    gated = run_block1c(page, layout, draw_debug=draw_debug)
    result = gated.to_result()
    result.extra = {**(result.extra or {}), "if1_block": "if1block1", "warp": warp_meta}
    return result


def normalize_batch(
    inputs: list | str | Path,
    *,
    output_dir: str | Path | None = None,
    output_zip: str | Path | None = None,
    template: TemplateSpec | Path | str | None = None,
    save_crops: bool = True,
) -> dict[str, Any]:
    """Same ZIP schema as Block 1; manifest block is ``if1block1``."""
    input_items = collect_image_inputs(inputs)
    if output_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix="if1block1_out_"))
    else:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    manifest_docs: list[dict[str, Any]] = []
    print(f"[if1block1] Starting batch for {len(input_items)} document(s)...")
    for doc_id, item in input_items:
        try:
            print(f"  → if1block1 '{doc_id}'...")
            res = normalize_document(item, template=template, document_id=doc_id)
            doc_out_dir = target_dir / "docs" / doc_id
            doc_meta = save_normalized_document(res, doc_out_dir, save_crops=save_crops)
            needs_review = bool(res.extra.get("needs_review"))
            status = "needs_review" if needs_review else "success"
            pick = res.extra.get("template_pick") or {}
            warp = res.extra.get("warp") or {}
            manifest_docs.append(
                {
                    "doc_id": doc_id,
                    "status": status,
                    "needs_review": needs_review,
                    "template_id": doc_meta["template_id"],
                    "n_checkbox": doc_meta.get("num_checkboxes"),
                    "template_pick": pick if isinstance(pick, dict) else {},
                    "canvas_size": doc_meta["canvas_size"],
                    "alignment_confidence": doc_meta["alignment_confidence"],
                    "num_checkboxes": doc_meta["num_checkboxes"],
                    "num_handwriting": doc_meta["num_handwriting"],
                    "warp_method": warp.get("method"),
                    "metadata_path": f"docs/{doc_id}/metadata.json",
                    "canonical_path": f"docs/{doc_id}/canonical.png",
                    "overlay_path": f"docs/{doc_id}/overlay.png" if doc_meta["overlay_path"] else None,
                }
            )
        except Exception as exc:
            print(f"  [!] Error processing '{doc_id}': {exc}")
            manifest_docs.append({"doc_id": doc_id, "status": "error", "error": str(exc)})

    manifest = {
        "version": "1.0",
        "block": "if1block1",
        "total_documents": len(input_items),
        "successful_documents": sum(1 for d in manifest_docs if d.get("status") == "success"),
        "template_selection": batch_template_conflict(manifest_docs),
        "documents": manifest_docs,
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
    return {
        "manifest": manifest,
        "output_dir": str(target_dir),
        "output_zip": str(zip_path) if zip_path else None,
    }

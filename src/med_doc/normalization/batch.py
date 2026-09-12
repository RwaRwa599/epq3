"""Batch normalization and standardized ZIP packaging for Block 1."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Sequence
import cv2
import numpy as np
from PIL import Image

from med_doc.normalization.pipeline import normalize_document
from med_doc.schemas import NormalizedDocumentResult, TemplateSpec


def _compute_mark_heuristic(crop_img: np.ndarray) -> tuple[float, bool]:
    """Interior dark ratio as registration-quality debug — not a tick label.

    Block 1 does not classify marks. Callers that still read
    ``is_marked_candidate`` should treat it as debug-only.
    """
    if crop_img.ndim == 3:
        gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop_img

    dark_ratio = float((gray < 130).mean())
    is_candidate = bool(dark_ratio >= 0.12)
    return round(dark_ratio, 4), is_candidate


def save_normalized_document(
    result: NormalizedDocumentResult,
    doc_dir: str | Path,
    *,
    save_crops: bool = True,
) -> dict[str, Any]:
    """Save a NormalizedDocumentResult into a standardized directory structure."""
    doc_path = Path(doc_dir)
    doc_path.mkdir(parents=True, exist_ok=True)

    # 1. Save canonical image
    can_bgr = cv2.cvtColor(result.canonical_canvas, cv2.COLOR_RGB2BGR)
    can_rel_path = "canonical.png"
    cv2.imwrite(str(doc_path / can_rel_path), can_bgr)

    # 2. Save overlay image
    overlay_rel_path = None
    if result.debug_overlay is not None:
        ov_bgr = cv2.cvtColor(result.debug_overlay, cv2.COLOR_RGB2BGR)
        overlay_rel_path = "overlay.png"
        cv2.imwrite(str(doc_path / overlay_rel_path), ov_bgr)

    tid = str(result.extra.get("template_id", ""))
    from med_doc.normalization.viz import draw_sections
    from med_doc.paths import DEFAULT_TEMPLATE, V1_TEMPLATE
    from med_doc.template import load_template

    spec = load_template(V1_TEMPLATE if "v1" in tid else DEFAULT_TEMPLATE)
    raw_sections = result.extra.get("sections")
    if raw_sections:
        from med_doc.schemas import SectionSpec

        spec = spec.model_copy(
            update={"sections": [SectionSpec.model_validate(item) for item in raw_sections]}
        )
    sections_vis = draw_sections(result.canonical_canvas, spec)
    cv2.imwrite(
        str(doc_path / "sections.png"),
        cv2.cvtColor(sections_vis, cv2.COLOR_RGB2BGR),
    )

    # 3. Save crops
    cb_crops_dir = doc_path / "crops" / "checkboxes"
    hw_crops_dir = doc_path / "crops" / "handwriting"
    sec_crops_dir = doc_path / "crops" / "sections"
    if save_crops:
        cb_crops_dir.mkdir(parents=True, exist_ok=True)
        hw_crops_dir.mkdir(parents=True, exist_ok=True)
        sec_crops_dir.mkdir(parents=True, exist_ok=True)

    fields_meta: dict[str, Any] = {"checkboxes": {}, "handwriting": {}, "sections": {}}
    detected_marks: dict[str, Any] = {}
    registration = result.extra.get("registration") or {}

    for fid, crop in result.checkbox_crops.items():
        dark_ratio, is_candidate = _compute_mark_heuristic(crop.normalized_image)
        # Kept for Block 3 ingest compatibility; not a Block 1 product signal.
        detected_marks[fid] = {
            "dark_ratio": dark_ratio,
            "is_marked_candidate": is_candidate,
            "debug": True,
        }

        crop_rel_path = None
        if save_crops:
            crop_rel_path = f"crops/checkboxes/{fid}.png"
            norm_bgr = cv2.cvtColor(crop.normalized_image, cv2.COLOR_RGB2BGR) if crop.normalized_image.ndim == 3 else crop.normalized_image
            cv2.imwrite(str(doc_path / crop_rel_path), norm_bgr)

        fields_meta["checkboxes"][fid] = {
            "bbox": crop.canonical_bbox,
            "quality_score": round(crop.quality_score, 3),
            "dark_ratio": dark_ratio,
            "crop_path": crop_rel_path,
            "crop_ok": bool(crop.crop_ok),
            "crop_needs_hitl": bool(crop.crop_needs_hitl),
            "crop_validate_status": crop.crop_validate_status,
            "crop_validate_attempts": int(crop.crop_validate_attempts),
            "debug": {"is_marked_candidate": is_candidate},
        }

    for fid, crop in result.handwriting_crops.items():
        crop_rel_path = None
        if save_crops:
            crop_rel_path = f"crops/handwriting/{fid}.png"
            norm_bgr = cv2.cvtColor(crop.normalized_image, cv2.COLOR_RGB2BGR) if crop.normalized_image.ndim == 3 else crop.normalized_image
            cv2.imwrite(str(doc_path / crop_rel_path), norm_bgr)

        fields_meta["handwriting"][fid] = {
            "bbox": crop.canonical_bbox,
            "quality_score": round(crop.quality_score, 3),
            "crop_path": crop_rel_path,
        }

    for sid, crop in result.section_crops.items():
        crop_rel_path = None
        if save_crops:
            crop_rel_path = f"crops/sections/{sid}.png"
            raw = crop.raw_image
            raw_bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR) if raw.ndim == 3 else raw
            cv2.imwrite(str(doc_path / crop_rel_path), raw_bgr)

        fields_meta["sections"][sid] = {
            "bbox": crop.canonical_bbox,
            "quality_score": round(crop.quality_score, 3),
            "field_ids": list(crop.field_ids),
            "crop_path": crop_rel_path,
        }

    # 4. Save metadata.json
    doc_meta = {
        "doc_id": result.document_id,
        "template_id": result.extra.get("template_id", "v1"),
        "canvas_size": [int(result.canonical_canvas.shape[1]), int(result.canonical_canvas.shape[0])],
        "alignment_confidence": round(result.alignment_confidence, 3),
        "warp_method": result.warp_method,
        "orientation_degrees": result.orientation_degrees,
        "num_checkboxes": len(result.checkbox_crops),
        "num_handwriting": len(result.handwriting_crops),
        "num_sections": len(result.section_crops),
        "canonical_path": can_rel_path,
        "overlay_path": overlay_rel_path,
        "detected_marks": detected_marks,
        "registration": registration,
        "crop_validate": result.extra.get("crop_validate") or {},
        "mark_classification": "deferred_to_block3",
        "fields": fields_meta,
        "extra": {k: v for k, v in result.extra.items() if isinstance(v, (str, int, float, bool, list, dict))},
    }

    (doc_path / "metadata.json").write_text(json.dumps(doc_meta, indent=2))
    return doc_meta


def normalize_batch(
    inputs: Sequence[str | Path | Image.Image | np.ndarray] | str | Path,
    *,
    output_dir: str | Path | None = None,
    output_zip: str | Path | None = None,
    template: TemplateSpec | Path | str | None = None,
    save_crops: bool = True,
) -> dict[str, Any]:
    """Normalize a batch of document images and optionally package into a standardized ZIP."""
    # Resolve input list
    input_items: list[tuple[str, Any]] = []
    
    if isinstance(inputs, (str, Path)):
        inp_path = Path(inputs)
        if inp_path.is_file() and inp_path.suffix.lower() == ".zip":
            # Input is a ZIP of raw images
            temp_in = tempfile.mkdtemp(prefix="raw_batch_")
            with zipfile.ZipFile(inp_path, "r") as z:
                z.extractall(temp_in)
            for p in sorted(Path(temp_in).rglob("*")):
                if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]:
                    input_items.append((p.stem, p))
        elif inp_path.is_dir():
            for p in sorted(inp_path.iterdir()):
                if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]:
                    input_items.append((p.stem, p))
        elif inp_path.is_file():
            input_items.append((inp_path.stem, inp_path))
    else:
        for idx, item in enumerate(inputs):
            if isinstance(item, (str, Path)):
                p = Path(item)
                input_items.append((p.stem, p))
            else:
                input_items.append((f"doc_{idx+1:03d}", item))

    if not input_items:
        raise ValueError(f"No valid image files found in input: {inputs}")

    # Prepare target directory
    is_temp_out = False
    if output_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix="block1_out_"))
        is_temp_out = True
    else:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    manifest_docs = []

    print(f"[Block 1] Starting batch normalization for {len(input_items)} document(s)...")

    for doc_id, item in input_items:
        try:
            print(f"  → Processing '{doc_id}'...")
            res = normalize_document(item, template=template, document_id=doc_id)
            doc_out_dir = target_dir / "docs" / doc_id
            doc_meta = save_normalized_document(res, doc_out_dir, save_crops=save_crops)
            
            manifest_docs.append({
                "doc_id": doc_id,
                "status": "success",
                "template_id": doc_meta["template_id"],
                "canvas_size": doc_meta["canvas_size"],
                "alignment_confidence": doc_meta["alignment_confidence"],
                "num_checkboxes": doc_meta["num_checkboxes"],
                "num_handwriting": doc_meta["num_handwriting"],
                "metadata_path": f"docs/{doc_id}/metadata.json",
                "canonical_path": f"docs/{doc_id}/canonical.png",
                "overlay_path": f"docs/{doc_id}/overlay.png" if doc_meta["overlay_path"] else None,
            })
        except Exception as e:
            print(f"  [!] Error processing '{doc_id}': {e}")
            manifest_docs.append({
                "doc_id": doc_id,
                "status": "error",
                "error": str(e),
            })

    manifest = {
        "version": "1.0",
        "block": "block1",
        "total_documents": len(input_items),
        "successful_documents": sum(1 for d in manifest_docs if d.get("status") == "success"),
        "documents": manifest_docs,
    }

    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # If ZIP output requested
    if output_zip is not None:
        zip_path = Path(output_zip)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Block 1] Packaging {len(manifest_docs)} processed document(s) into '{zip_path}'...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in target_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(target_dir)
                    zf.write(file_path, arcname=arcname)
        print(f"✓ ZIP bundle created successfully: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB)")

    return {
        "manifest": manifest,
        "output_dir": str(target_dir),
        "output_zip": str(output_zip) if output_zip else None,
    }

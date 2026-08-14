"""Batch processing and Block 1 -> Block 2 -> Block 3 pipeline integration."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from med_doc.kg.graph import KnowledgeGraph
from med_doc.kg.schemas import ValidationResult


def process_batch_from_block1(
    input_source: str | Path,
    *,
    output_dir: str | Path | None = None,
    output_zip: str | Path | None = None,
    kg: KnowledgeGraph | None = None,
    dark_ratio_threshold: float = 0.12,
) -> dict[str, Any]:
    """Ingest Block 1 ZIP/folder, apply Knowledge Graph priors & validation, and emit Block 3 ZIP."""
    if kg is None:
        kg = KnowledgeGraph.load()

    input_path = Path(input_source)
    is_temp_in = False

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        temp_in = Path(tempfile.mkdtemp(prefix="block1_in_"))
        print(f"[Block 2] Extracting input ZIP: {input_path.name}...")
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(temp_in)
        base_in_dir = temp_in
        is_temp_in = True
    elif input_path.is_dir():
        base_in_dir = input_path
    else:
        raise ValueError(f"Invalid input source: {input_source}")

    manifest_file = base_in_dir / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Missing 'manifest.json' in Block 1 data at {base_in_dir}")

    b1_manifest = json.loads(manifest_file.read_text())
    print(f"[Block 2] Ingested Block 1 manifest: {b1_manifest.get('total_documents', 0)} document(s)")

    if output_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix="block2_out_"))
    else:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    b2_manifest_docs = []

    for doc_entry in b1_manifest.get("documents", []):
        doc_id = doc_entry["doc_id"]
        if doc_entry.get("status") == "error":
            print(f"  [!] Skipping failed Block 1 doc '{doc_id}'")
            continue

        print(f"  → Validating & Applying Priors for '{doc_id}'...")
        doc_dir_in = base_in_dir / "docs" / doc_id
        doc_meta_file = doc_dir_in / "metadata.json"
        if not doc_meta_file.exists():
            print(f"    [!] Warning: missing metadata.json for {doc_id}")
            continue

        doc_meta = json.loads(doc_meta_file.read_text())
        detected_marks = doc_meta.get("detected_marks", {})

        # Extract ticked checkboxes
        ticked_ids = []
        for fid, mark_info in detected_marks.items():
            if mark_info.get("is_marked_candidate") or mark_info.get("dark_ratio", 0) >= dark_ratio_threshold:
                ticked_ids.append(fid)

        # Prepare target doc directory
        doc_dir_out = target_dir / "docs" / doc_id
        doc_dir_out.mkdir(parents=True, exist_ok=True)

        # Copy canonical image
        can_in = doc_dir_in / "canonical.png"
        if can_in.exists():
            shutil.copy2(can_in, doc_dir_out / "canonical.png")

        # Copy overlay image if present
        ov_in = doc_dir_in / "overlay.png"
        if ov_in.exists():
            shutil.copy2(ov_in, doc_dir_out / "overlay.png")

        # Copy checkbox + handwriting crops (needed for downstream Block 3 HTR)
        for sub in ("checkboxes", "handwriting"):
            src_dir = doc_dir_in / "crops" / sub
            if src_dir.exists():
                dst_dir = doc_dir_out / "crops" / sub
                dst_dir.mkdir(parents=True, exist_ok=True)
                for crop in src_dir.glob("*.png"):
                    shutil.copy2(crop, dst_dir / crop.name)

        # 1. Expand Profiles & Calculate expected tubes
        implied = kg.implied_tests(ticked_ids)
        expected_tubes = kg.calculate_expected_tubes(ticked_ids)

        # 2. Run assume() prior rankings for handwriting fields (e.g. 'others')
        prior_rankings: dict[str, Any] = {}
        context = {"ticked_ids": ticked_ids}

        hw_fields = doc_meta.get("fields", {}).get("handwriting", {})
        for hw_fid in hw_fields.keys():
            # Example default query for others
            if hw_fid == "others":
                candidates = kg.assume("others", "culture", context=context, top_k=3)
                prior_rankings[hw_fid] = [c.model_dump() for c in candidates]
            elif hw_fid.startswith("tube_"):
                tube_name = kg.tube_field_to_tube.get(hw_fid, "")
                exp_c = expected_tubes.get(tube_name, 1)
                candidates = kg.assume(hw_fid, str(exp_c), context=context)
                prior_rankings[hw_fid] = [c.model_dump() for c in candidates]

        # 3. Clinical cross-field validation
        # By default assume observed tubes match expected unless nurse count detected
        observed_tubes = {t: c for t, c in expected_tubes.items()}
        val_report = kg.validate_request(
            ticked_ids=ticked_ids,
            observed_tubes=observed_tubes,
        )

        # Save validation report and prior rankings
        (doc_dir_out / "validation_report.json").write_text(val_report.model_dump_json(indent=2))
        (doc_dir_out / "prior_rankings.json").write_text(json.dumps(prior_rankings, indent=2))
        (doc_dir_out / "metadata.json").write_text(json.dumps(doc_meta, indent=2))

        b2_manifest_docs.append({
            "doc_id": doc_id,
            "is_valid": val_report.is_valid,
            "confidence": val_report.confidence,
            "ticked_tests": ticked_ids,
            "implied_tests": sorted(implied),
            "expected_tubes": expected_tubes,
            "discrepancies": val_report.discrepancies,
            "warnings": val_report.warnings,
            "validation_report_path": f"docs/{doc_id}/validation_report.json",
            "prior_rankings_path": f"docs/{doc_id}/prior_rankings.json",
            "canonical_path": f"docs/{doc_id}/canonical.png",
            "metadata_path": f"docs/{doc_id}/metadata.json",
            "handwriting_crops_dir": f"docs/{doc_id}/crops/handwriting",
            "checkbox_crops_dir": f"docs/{doc_id}/crops/checkboxes",
        })

    b2_manifest = {
        "version": "1.0",
        "block": "block2",
        "stage": "clinical_validation_and_priors",
        "total_documents": len(b2_manifest_docs),
        "valid_documents": sum(1 for d in b2_manifest_docs if d["is_valid"]),
        "documents": b2_manifest_docs,
    }

    (target_dir / "manifest.json").write_text(json.dumps(b2_manifest, indent=2))

    # Package output ZIP for Block 3
    if output_zip is not None:
        zip_path = Path(output_zip)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Block 2] Packaging Block 3 bundle into '{zip_path}'...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in target_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(target_dir)
                    zf.write(file_path, arcname=arcname)
        print(f"✓ Block 2 -> Block 3 ZIP created: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB)")

    if is_temp_in:
        shutil.rmtree(temp_in, ignore_errors=True)

    return {
        "manifest": b2_manifest,
        "output_dir": str(target_dir),
        "output_zip": str(output_zip) if output_zip else None,
    }

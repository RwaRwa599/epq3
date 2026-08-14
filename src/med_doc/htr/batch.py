"""Batch HTR: ingest Block 2 ZIP, classify marks, fuse handwriting, export Block 3 ZIP."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from med_doc.htr.fusion import fuse_handwriting
from med_doc.htr.marks import classify_mark
from med_doc.htr.recognizer import extract_digits, recognize_handwriting
from med_doc.htr.schemas import BatchPredictionManifest, DocumentPrediction, HandwritingPrediction, MarkPrediction
from med_doc.htr.viz import draw_prediction_overlay
from med_doc.kg.graph import KnowledgeGraph


def _load_rgb(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _unzip_or_dir(input_source: str | Path) -> tuple[Path, bool]:
    input_path = Path(input_source)
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        temp_in = Path(tempfile.mkdtemp(prefix="block2_in_"))
        print(f"[Block 3] Extracting input ZIP: {input_path.name}...")
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(temp_in)
        return temp_in, True
    if input_path.is_dir():
        return input_path, False
    raise ValueError(f"Invalid Block 2 input source: {input_source}")


def _observed_tubes_from_hw(
    hw_fields: dict[str, HandwritingPrediction],
    kg: KnowledgeGraph | None,
) -> dict[str, int | None]:
    observed: dict[str, int | None] = {}
    mapping = kg.tube_field_to_tube if kg is not None else {}
    for fid, pred in hw_fields.items():
        if not fid.startswith("tube_"):
            continue
        tube = mapping.get(fid, fid.replace("tube_", "").upper())
        digits = extract_digits(pred.canonical_value or pred.raw_text or "")
        observed[tube] = int(digits) if digits else None
    return observed


def process_document(
    doc_dir: Path,
    *,
    kg: KnowledgeGraph | None = None,
    backend: str = "auto",
) -> DocumentPrediction:
    """Run mark classification + HTR fusion for a single Block 2 document folder."""
    meta_path = doc_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {doc_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    doc_id = str(meta.get("doc_id") or doc_dir.name)

    priors_path = doc_dir / "prior_rankings.json"
    prior_rankings: dict[str, Any] = {}
    if priors_path.exists():
        prior_rankings = json.loads(priors_path.read_text(encoding="utf-8"))

    val_path = doc_dir / "validation_report.json"
    val_report: dict[str, Any] = {}
    if val_path.exists():
        val_report = json.loads(val_path.read_text(encoding="utf-8"))

    fields_meta = meta.get("fields", {})
    detected_marks = meta.get("detected_marks", {})
    cb_meta = fields_meta.get("checkboxes", {})
    hw_meta = fields_meta.get("handwriting", {})

    checkbox_marks: dict[str, MarkPrediction] = {}
    ticked_ids: list[str] = []

    field_ids = list(cb_meta.keys()) or list(detected_marks.keys())
    for fid in field_ids:
        crop_rel = (cb_meta.get(fid) or {}).get("crop_path") or f"crops/checkboxes/{fid}.png"
        crop = _load_rgb(doc_dir / crop_rel)
        mark_info = detected_marks.get(fid, {})
        pred = classify_mark(
            crop,
            fid,
            fallback_dark_ratio=mark_info.get("dark_ratio"),
            fallback_candidate=mark_info.get("is_marked_candidate"),
        )
        checkbox_marks[fid] = pred
        if pred.is_marked:
            ticked_ids.append(fid)

    expected_tubes: dict[str, int] = {}
    implied: list[str] = []
    if kg is not None:
        expected_tubes = kg.calculate_expected_tubes(ticked_ids)
        implied = sorted(kg.implied_tests(ticked_ids))
    else:
        expected_tubes = dict(val_report.get("expected_tubes") or {})
        implied = list(val_report.get("implied_tests") or [])

    handwriting_fields: dict[str, HandwritingPrediction] = {}
    hw_ids = list(hw_meta.keys()) or list(prior_rankings.keys())
    for fid in hw_ids:
        crop_rel = (hw_meta.get(fid) or {}).get("crop_path") or f"crops/handwriting/{fid}.png"
        crop = _load_rgb(doc_dir / crop_rel)
        draft = recognize_handwriting(crop, fid, backend=backend)
        fused = fuse_handwriting(
            fid,
            draft.text,
            draft.confidence,
            draft.source,
            prior_rankings=prior_rankings.get(fid),
            kg=kg,
            ticked_ids=ticked_ids,
            expected_tubes=expected_tubes,
        )
        handwriting_fields[fid] = fused

    observed_tubes = _observed_tubes_from_hw(handwriting_fields, kg)
    discrepancies: list[str] = []
    warnings: list[str] = []
    is_valid = True
    if kg is not None:
        report = kg.validate_request(
            ticked_ids=ticked_ids,
            observed_tubes=observed_tubes,
            write_ins=[
                (p.canonical_value or p.raw_text or "")
                for fid, p in handwriting_fields.items()
                if fid == "others" and (p.canonical_value or p.raw_text)
            ],
        )
        discrepancies = report.discrepancies
        warnings = report.warnings
        is_valid = report.is_valid
        expected_tubes = report.expected_tubes
        implied = report.implied_tests
    else:
        discrepancies = list(val_report.get("discrepancies") or [])
        warnings = list(val_report.get("warnings") or [])
        is_valid = bool(val_report.get("is_valid", True))

    hitl_fields = [
        fid for fid, m in checkbox_marks.items() if m.needs_hitl
    ] + [
        fid for fid, h in handwriting_fields.items() if h.needs_hitl
    ]

    confs = [m.confidence for m in checkbox_marks.values()] + [
        h.confidence for h in handwriting_fields.values()
    ]
    overall = float(sum(confs) / len(confs)) if confs else 1.0
    if hitl_fields:
        overall = min(overall, 0.74)
    if discrepancies:
        overall = min(overall, 0.6)

    return DocumentPrediction(
        doc_id=doc_id,
        is_valid=is_valid and not discrepancies,
        checkbox_marks=checkbox_marks,
        handwriting_fields=handwriting_fields,
        ticked_test_ids=ticked_ids,
        implied_tests=implied,
        expected_tubes=expected_tubes,
        observed_tubes=observed_tubes,
        discrepancies=discrepancies,
        warnings=warnings,
        overall_confidence=round(overall, 3),
        hitl_fields=hitl_fields,
    )


def process_batch_from_block2(
    input_source: str | Path,
    *,
    output_dir: str | Path | None = None,
    output_zip: str | Path | None = None,
    kg: KnowledgeGraph | None = None,
    backend: str = "auto",
) -> dict[str, Any]:
    """Ingest Block 2 ZIP/folder, run HTR + prior fusion, emit Block 3 ZIP."""
    if kg is None:
        try:
            kg = KnowledgeGraph.load()
        except FileNotFoundError:
            kg = None

    base_in_dir, is_temp_in = _unzip_or_dir(input_source)
    manifest_file = base_in_dir / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Missing 'manifest.json' in Block 2 data at {base_in_dir}")

    b2_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    print(f"[Block 3] Ingested Block 2 manifest: {b2_manifest.get('total_documents', 0)} document(s)")

    if output_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix="block3_out_"))
    else:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    docs_out: list[dict[str, Any]] = []

    for doc_entry in b2_manifest.get("documents", []):
        doc_id = doc_entry["doc_id"]
        if doc_entry.get("status") == "error":
            print(f"  [!] Skipping failed upstream doc '{doc_id}'")
            continue

        print(f"  → HTR + prior fusion for '{doc_id}'...")
        doc_dir_in = base_in_dir / "docs" / doc_id
        if not (doc_dir_in / "metadata.json").exists():
            print(f"    [!] Warning: missing metadata.json for {doc_id}")
            continue

        prediction = process_document(doc_dir_in, kg=kg, backend=backend)

        doc_dir_out = target_dir / "docs" / doc_id
        doc_dir_out.mkdir(parents=True, exist_ok=True)

        # Carry forward canonical + metadata + crops for Block 4
        for name in ("canonical.png", "overlay.png", "metadata.json", "validation_report.json", "prior_rankings.json"):
            src = doc_dir_in / name
            if src.exists():
                shutil.copy2(src, doc_dir_out / name)

        for sub in ("crops/checkboxes", "crops/handwriting"):
            src_dir = doc_dir_in / sub
            if src_dir.exists():
                dst_dir = doc_dir_out / sub
                dst_dir.mkdir(parents=True, exist_ok=True)
                for crop in src_dir.glob("*.png"):
                    shutil.copy2(crop, dst_dir / crop.name)

        (doc_dir_out / "prediction.json").write_text(
            prediction.model_dump_json(indent=2), encoding="utf-8"
        )

        canvas = _load_rgb(doc_dir_in / "canonical.png")
        meta = json.loads((doc_dir_in / "metadata.json").read_text(encoding="utf-8"))
        if canvas is not None:
            overlay = draw_prediction_overlay(canvas, prediction, meta.get("fields", {}))
            cv2.imwrite(
                str(doc_dir_out / "annotated_canvas.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

        docs_out.append(
            {
                "doc_id": doc_id,
                "is_valid": prediction.is_valid,
                "overall_confidence": prediction.overall_confidence,
                "n_ticked": len(prediction.ticked_test_ids),
                "n_hitl": len(prediction.hitl_fields),
                "hitl_fields": prediction.hitl_fields,
                "ticked_test_ids": prediction.ticked_test_ids,
                "expected_tubes": prediction.expected_tubes,
                "discrepancies": prediction.discrepancies,
                "prediction_path": f"docs/{doc_id}/prediction.json",
                "annotated_canvas_path": f"docs/{doc_id}/annotated_canvas.png",
            }
        )

    manifest = BatchPredictionManifest(
        total_documents=len(docs_out),
        valid_documents=sum(1 for d in docs_out if d["is_valid"]),
        hitl_documents=sum(1 for d in docs_out if d["n_hitl"] > 0),
        documents=docs_out,
    )
    (target_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    if output_zip is not None:
        zip_path = Path(output_zip)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Block 3] Packaging Block 4 bundle into '{zip_path}'...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in target_dir.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, arcname=file_path.relative_to(target_dir))
        print(
            f"✓ Block 3 -> Block 4 ZIP created: {zip_path} "
            f"({zip_path.stat().st_size / 1024 / 1024:.2f} MB)"
        )

    if is_temp_in:
        shutil.rmtree(base_in_dir, ignore_errors=True)

    return {
        "manifest": manifest.model_dump(),
        "output_dir": str(target_dir),
        "output_zip": str(output_zip) if output_zip else None,
    }

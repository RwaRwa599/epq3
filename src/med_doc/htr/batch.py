"""Block 3 batch: ingest Block 1 ZIP, run nonverbal and/or verbal, emit hypotheses."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import cv2

from med_doc.htr.ingest import (
    Block1Document,
    cleanup_temp,
    iter_block1_documents,
    load_rgb,
    unzip_or_dir,
)
from med_doc.htr.nonverbal import classify_marks
from med_doc.htr.schemas import (
    BatchPredictionManifest,
    DocumentHypotheses,
    DocumentPrediction,
    HandwritingPrediction,
    MarkPrediction,
)
from med_doc.htr.verbal import recognize_fields
from med_doc.htr.viz import draw_prediction_overlay
from med_doc.kg.graph import KnowledgeGraph

Mode = Literal["nonverbal", "verbal", "both"]


def _ticked_ids(marks: dict[str, MarkPrediction]) -> list[str]:
    return [fid for fid, pred in marks.items() if pred.is_marked]


def _overall(marks: dict[str, MarkPrediction], hw: dict[str, HandwritingPrediction]) -> float:
    confs = [m.confidence for m in marks.values()] + [h.confidence for h in hw.values()]
    return float(sum(confs) / len(confs)) if confs else 1.0


def _hitl(marks: dict[str, MarkPrediction], hw: dict[str, HandwritingPrediction]) -> list[str]:
    return [fid for fid, m in marks.items() if m.needs_hitl] + [
        fid for fid, h in hw.items() if h.needs_hitl
    ]


def hypotheses_from_parts(
    doc_id: str,
    marks: dict[str, MarkPrediction],
    hw: dict[str, HandwritingPrediction],
    *,
    implied: list[str] | None = None,
) -> DocumentHypotheses:
    ticked = _ticked_ids(marks)
    hitl = _hitl(marks, hw)
    overall = _overall(marks, hw)
    if hitl:
        overall = min(overall, 0.74)
    return DocumentHypotheses(
        doc_id=doc_id,
        nonverbal=marks,
        verbal=hw,
        ticked_test_ids=ticked,
        implied_tests=list(implied or []),
        overall_confidence=round(overall, 3),
        hitl_fields=hitl,
    )


def prediction_from_hypotheses(
    hyp: DocumentHypotheses,
    *,
    kg: KnowledgeGraph | None = None,
    observed_tubes: dict[str, int | None] | None = None,
) -> DocumentPrediction:
    expected: dict[str, int] = {}
    implied = list(hyp.implied_tests)
    if kg is not None:
        expected = kg.calculate_expected_tubes(hyp.ticked_test_ids)
        implied = sorted(kg.implied_tests(hyp.ticked_test_ids))
    return DocumentPrediction(
        doc_id=hyp.doc_id,
        checkbox_marks=hyp.nonverbal,
        handwriting_fields=hyp.verbal,
        ticked_test_ids=hyp.ticked_test_ids,
        implied_tests=implied,
        expected_tubes=expected,
        observed_tubes=observed_tubes or {},
        overall_confidence=hyp.overall_confidence,
        hitl_fields=hyp.hitl_fields,
    )


def process_document(
    doc_dir: Path,
    *,
    kg: KnowledgeGraph | None = None,
    backend: str = "auto",
    mode: Mode = "both",
) -> DocumentPrediction:
    """Run nonverbal and/or verbal recognition for one Block 1 document folder."""
    from med_doc.htr.ingest import load_block1_document

    found = load_block1_document(doc_dir)
    hyp = process_block1_document(found, kg=kg, backend=backend, mode=mode)
    return prediction_from_hypotheses(hyp, kg=kg)


def process_block1_document(
    doc: Block1Document,
    *,
    kg: KnowledgeGraph | None = None,
    backend: str = "auto",
    mode: Mode = "both",
) -> DocumentHypotheses:
    """Nonverbal and/or verbal on one ingested Block 1 document."""
    run_nv = mode in ("nonverbal", "both")
    run_vb = mode in ("verbal", "both")

    marks: dict[str, MarkPrediction] = {}
    if run_nv:
        marks = classify_marks(doc.checkbox_crops, fallbacks=doc.detected_marks)

    # KG scoring (assume / tubes / implied tests) is Block 4. Block 3 emits drafts only.
    hw: dict[str, HandwritingPrediction] = {}
    if run_vb:
        verbal_backend = "trocr" if backend in ("auto", "trocr") else backend
        if backend == "lexicon":
            verbal_backend = "auto"
        hw = recognize_fields(doc.handwriting_crops, backend=verbal_backend)

    return hypotheses_from_parts(doc.doc_id, marks, hw)


def process_from_block1(
    input_source: str | Path,
    *,
    output_dir: str | Path | None = None,
    output_zip: str | Path | None = None,
    kg: KnowledgeGraph | None = None,
    backend: str = "auto",
    mode: Mode = "both",
) -> dict[str, Any]:
    """Ingest Block 1 ZIP/folder, run verbal/nonverbal, emit hypotheses.json + ZIP.

    Does not require Block 2's per-sheet batch. `kg` is accepted for API compatibility
    but is not applied here — Block 4 runs `assume()` / tube constraints on these drafts.
    """
    _ = kg  # Block 4; do not auto-load or fuse in Block 3.

    base_in_dir, is_temp_in = unzip_or_dir(input_source)
    print(f"[Block 3] Ingested {input_source} (mode={mode})")

    if output_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix="block3_out_"))
    else:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    docs_out: list[dict[str, Any]] = []

    for doc in iter_block1_documents(base_in_dir):
        print(f"  → {mode} recognition for '{doc.doc_id}'...")
        hyp = process_block1_document(doc, kg=None, backend=backend, mode=mode)
        prediction = prediction_from_hypotheses(hyp, kg=None)

        doc_dir_out = target_dir / "docs" / doc.doc_id
        doc_dir_out.mkdir(parents=True, exist_ok=True)

        for name in ("canonical.png", "overlay.png", "metadata.json"):
            src = doc.doc_dir / name
            if src.exists():
                shutil.copy2(src, doc_dir_out / name)

        for sub in ("crops/checkboxes", "crops/handwriting"):
            src_dir = doc.doc_dir / sub
            if src_dir.exists():
                dst_dir = doc_dir_out / sub
                dst_dir.mkdir(parents=True, exist_ok=True)
                for crop in src_dir.glob("*.png"):
                    shutil.copy2(crop, dst_dir / crop.name)

        (doc_dir_out / "hypotheses.json").write_text(hyp.model_dump_json(indent=2), encoding="utf-8")
        (doc_dir_out / "prediction.json").write_text(
            prediction.model_dump_json(indent=2), encoding="utf-8"
        )

        canvas = load_rgb(doc.doc_dir / "canonical.png")
        if canvas is not None:
            overlay = draw_prediction_overlay(canvas, prediction, doc.fields)
            cv2.imwrite(
                str(doc_dir_out / "annotated_canvas.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

        docs_out.append(
            {
                "doc_id": doc.doc_id,
                "is_valid": prediction.is_valid,
                "overall_confidence": prediction.overall_confidence,
                "n_ticked": len(prediction.ticked_test_ids),
                "n_hitl": len(prediction.hitl_fields),
                "hitl_fields": prediction.hitl_fields,
                "ticked_test_ids": prediction.ticked_test_ids,
                "hypotheses_path": f"docs/{doc.doc_id}/hypotheses.json",
                "prediction_path": f"docs/{doc.doc_id}/prediction.json",
                "annotated_canvas_path": f"docs/{doc.doc_id}/annotated_canvas.png",
            }
        )

    manifest = BatchPredictionManifest(
        total_documents=len(docs_out),
        valid_documents=sum(1 for d in docs_out if d["is_valid"]),
        hitl_documents=sum(1 for d in docs_out if d["n_hitl"] > 0),
        documents=docs_out,
    )
    (target_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    zip_path: Path | None = None
    if output_zip is not None:
        zip_path = Path(output_zip)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Block 3] Packaging '{zip_path}'...")
        import zipfile

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in target_dir.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, arcname=file_path.relative_to(target_dir))
        print(f"✓ Block 3 ZIP created: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")

    cleanup_temp(base_in_dir, is_temp_in)

    return {
        "manifest": manifest.model_dump(),
        "output_dir": str(target_dir),
        "output_zip": str(zip_path) if zip_path else None,
    }


def process_batch_from_block2(
    input_source: str | Path,
    *,
    output_dir: str | Path | None = None,
    output_zip: str | Path | None = None,
    kg: KnowledgeGraph | None = None,
    backend: str = "auto",
) -> dict[str, Any]:
    """Compatibility alias: Block 2 ZIPs are a superset of Block 1 folders."""
    return process_from_block1(
        input_source,
        output_dir=output_dir,
        output_zip=output_zip,
        kg=kg,
        backend=backend,
        mode="both",
    )

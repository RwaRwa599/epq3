"""Block 4 batch: ingest Block 3 hypotheses ZIP, emit LIS prediction.json."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2

from med_doc.htr.ingest import cleanup_temp, load_rgb, unzip_or_dir
from med_doc.htr.schemas import BatchPredictionManifest, DocumentHypotheses
from med_doc.htr.viz import draw_prediction_overlay
from med_doc.kg.graph import KnowledgeGraph
from med_doc.paths import DEFAULT_KG
from med_doc.rescoring.engine import rescore_hypotheses
from med_doc.rescoring.combinations import CombinationCritic, make_critic


def _iter_block3_docs(base_dir: Path) -> list[Path]:
    docs_root = base_dir / "docs"
    if not docs_root.is_dir():
        return []
    found: list[Path] = []
    for doc_dir in sorted(p for p in docs_root.iterdir() if p.is_dir()):
        if (doc_dir / "hypotheses.json").exists():
            found.append(doc_dir)
    return found


def process_from_block3(
    input_source: str | Path,
    *,
    output_dir: str | Path | None = None,
    output_zip: str | Path | None = None,
    kg: KnowledgeGraph | None = None,
    combo_backend: str = "off",
    combo_model: str | None = None,
    combo_threshold: float | None = None,
    critic: CombinationCritic | None = None,
) -> dict[str, Any]:
    """Ingest a Block 3 ZIP/folder, rescore with the frozen KG, write prediction.json.

    Leaves hypotheses.json unchanged so OCR drafts stay auditable.
    Does not read Block 1 ``detected_marks`` / dark_ratio as ticks.
    ``combo_backend="ollama"`` uses a small Instruct model as a placeholder for
    Block 2 co-occurrence tables (flags review only).
    """
    kg = kg or KnowledgeGraph.load(DEFAULT_KG)
    if critic is None:
        critic = make_critic(combo_backend, model=combo_model)
    base_in_dir, is_temp_in = unzip_or_dir(input_source)
    print(f"[Block 4] Ingested {input_source}")

    if output_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix="block4_out_"))
    else:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    docs_out: list[dict[str, Any]] = []
    for doc_dir in _iter_block3_docs(base_in_dir):
        hyp = DocumentHypotheses.model_validate_json(
            (doc_dir / "hypotheses.json").read_text(encoding="utf-8")
        )
        print(f"  → rescoring '{hyp.doc_id}'...")
        crops: dict = {}
        cb_dir = doc_dir / "crops" / "checkboxes"
        if cb_dir.is_dir():
            for png in cb_dir.glob("*.png"):
                crops[png.stem] = load_rgb(png)
        kw: dict[str, Any] = {"critic": critic, "checkbox_crops": crops or None}
        if combo_threshold is not None:
            kw["combo_threshold"] = combo_threshold
        prediction = rescore_hypotheses(hyp, kg, **kw)

        doc_dir_out = target_dir / "docs" / hyp.doc_id
        doc_dir_out.mkdir(parents=True, exist_ok=True)
        for src in doc_dir.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(doc_dir)
            if rel.name in {"prediction.json", "annotated_canvas.png"}:
                continue
            dest = doc_dir_out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

        (doc_dir_out / "prediction.json").write_text(
            prediction.model_dump_json(indent=2), encoding="utf-8"
        )

        canvas = load_rgb(doc_dir_out / "canonical.png")
        meta_path = doc_dir_out / "metadata.json"
        if canvas is not None and meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            overlay = draw_prediction_overlay(canvas, prediction, meta.get("fields") or {})
            cv2.imwrite(
                str(doc_dir_out / "annotated_canvas.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

        docs_out.append(
            {
                "doc_id": prediction.doc_id,
                "is_valid": prediction.is_valid,
                "overall_confidence": prediction.overall_confidence,
                "n_ticked": len(prediction.ticked_test_ids),
                "n_hitl": len(prediction.hitl_fields),
                "hitl_fields": prediction.hitl_fields,
                "ticked_test_ids": prediction.ticked_test_ids,
                "expected_tubes": prediction.expected_tubes,
                "observed_tubes": prediction.observed_tubes,
                "discrepancies": prediction.discrepancies,
                "hypotheses_path": f"docs/{prediction.doc_id}/hypotheses.json",
                "prediction_path": f"docs/{prediction.doc_id}/prediction.json",
            }
        )

    manifest = BatchPredictionManifest(
        block="block4",
        stage="kg_rescoring",
        total_documents=len(docs_out),
        valid_documents=sum(1 for d in docs_out if d["is_valid"]),
        hitl_documents=sum(1 for d in docs_out if d["n_hitl"] > 0),
        documents=docs_out,
    )
    (target_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    in_manifest = base_in_dir / "manifest.json"
    if in_manifest.exists() and not (target_dir / "block3_manifest.json").exists():
        shutil.copy2(in_manifest, target_dir / "block3_manifest.json")

    zip_path: Path | None = None
    if output_zip is not None:
        zip_path = Path(output_zip)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Block 4] Packaging '{zip_path}'...")
        import zipfile

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in target_dir.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, arcname=file_path.relative_to(target_dir))
        print(f"✓ Block 4 ZIP created: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")

    cleanup_temp(base_in_dir, is_temp_in)

    return {
        "manifest": manifest.model_dump(),
        "output_dir": str(target_dir),
        "output_zip": str(zip_path) if zip_path else None,
    }

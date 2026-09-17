"""if1block4: coverage scorer → two-tier order + 3a re-pass on flags."""

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
from med_doc.if1 import if1block2
from med_doc.if1.scorer import flags_from_scores, split_tiers
from med_doc.kg.graph import KnowledgeGraph
from med_doc.paths import DEFAULT_KG
from med_doc.rescoring.combinations import rerun_3a_on_flags
from med_doc.rescoring.engine import rescore_hypotheses
from med_doc.rescoring.ticks import trusted_tick_ids


def _iter_block3_docs(base_dir: Path) -> list[Path]:
    docs_root = base_dir / "docs"
    if not docs_root.is_dir():
        return []
    return sorted(
        p for p in docs_root.iterdir() if p.is_dir() and (p / "hypotheses.json").exists()
    )


def process_from_if1block3(
    input_source: str | Path,
    *,
    output_dir: str | Path | None = None,
    output_zip: str | Path | None = None,
    kg: KnowledgeGraph | None = None,
) -> dict[str, Any]:
    kg = kg or KnowledgeGraph.load(DEFAULT_KG)
    groups = if1block2.load_groups(kg=kg)
    base_in_dir, is_temp_in = unzip_or_dir(input_source)
    print(f"[if1block4] Ingested {input_source}")

    if output_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix="if1block4_out_"))
    else:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    docs_out: list[dict[str, Any]] = []
    for doc_dir in _iter_block3_docs(base_in_dir):
        hyp = DocumentHypotheses.model_validate_json(
            (doc_dir / "hypotheses.json").read_text(encoding="utf-8")
        )
        initial = list(hyp.initial_ticked_test_ids or trusted_tick_ids(hyp.nonverbal))
        scored = if1block2.score(initial, groups=groups, kg=kg)
        flags = flags_from_scores(scored, initial)
        crops: dict = {}
        cb_dir = doc_dir / "crops" / "checkboxes"
        if cb_dir.is_dir():
            for png in cb_dir.glob("*.png"):
                crops[png.stem] = load_rgb(png)
        nonverbal = rerun_3a_on_flags(hyp, flags, crops or None)
        hyp_scored = hyp.model_copy(
            update={
                "nonverbal": nonverbal,
                "ticked_test_ids": list(initial),
                "initial_ticked_test_ids": list(initial),
            }
        )
        prediction = rescore_hypotheses(hyp_scored, kg, checkbox_crops=crops or None)
        still = trusted_tick_ids(nonverbal)
        tiers = split_tiers(initial, scored, still_ticked=still)
        warnings = list(prediction.warnings)
        for fid in tiers["ordered_tests_low"]:
            line = f"if1 low-confidence test (not LIS-high): {fid}"
            if line not in warnings:
                warnings.append(line)
        prediction = prediction.model_copy(
            update={
                "initial_ticked_test_ids": list(initial),
                "ticked_test_ids": list(tiers["ordered_tests_high"]),
                "ordered_tests_high": list(tiers["ordered_tests_high"]),
                "ordered_tests_low": list(tiers["ordered_tests_low"]),
                "combo_scores": {
                    "m": scored.get("m"),
                    "e": scored.get("e"),
                    "activations": scored.get("activations"),
                },
                "warnings": warnings,
            }
        )

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
        # Frozen 3a snapshot on copied hypotheses.
        copied_hyp = DocumentHypotheses.model_validate_json(
            (doc_dir_out / "hypotheses.json").read_text(encoding="utf-8")
        )
        copied_hyp = copied_hyp.model_copy(update={"initial_ticked_test_ids": list(initial)})
        (doc_dir_out / "hypotheses.json").write_text(
            copied_hyp.model_dump_json(indent=2), encoding="utf-8"
        )
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
                "n_ticked": len(prediction.ticked_test_ids),
                "ordered_tests_high": prediction.ordered_tests_high,
                "ordered_tests_low": prediction.ordered_tests_low,
                "initial_ticked_test_ids": prediction.initial_ticked_test_ids,
                "prediction_path": f"docs/{prediction.doc_id}/prediction.json",
            }
        )

    manifest = BatchPredictionManifest(
        block="if1block4",
        stage="if1_coverage_rescoring",
        total_documents=len(docs_out),
        valid_documents=sum(1 for d in docs_out if d.get("is_valid")),
        documents=docs_out,
    )
    (target_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    zip_path = None
    if output_zip is not None:
        zip_path = Path(output_zip)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        import zipfile

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in target_dir.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, arcname=file_path.relative_to(target_dir))

    cleanup_temp(base_in_dir, is_temp_in)
    return {
        "manifest": manifest.model_dump(),
        "output_dir": str(target_dir),
        "output_zip": str(zip_path) if zip_path else None,
    }

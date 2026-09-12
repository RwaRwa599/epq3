"""Block 5 batch: ingest Block 4 ZIP, HiTL queue, optional patches, LIS order."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from med_doc.htr.ingest import cleanup_temp, unzip_or_dir
from med_doc.htr.schemas import BatchPredictionManifest, DocumentHypotheses, DocumentPrediction
from med_doc.kg.graph import KnowledgeGraph
from med_doc.paths import DEFAULT_KG
from med_doc.rescoring.engine import rescore_hypotheses
from med_doc.review.apply import commit_hypotheses
from med_doc.review.lis import order_from_prediction
from med_doc.review.llm import LlmRanker, attach_llm_suggestions
from med_doc.review.queue import build_hitl_queue
from med_doc.review.schemas import DocumentReview, ReviewPatch


def _iter_docs(base_dir: Path) -> list[Path]:
    docs_root = base_dir / "docs"
    if not docs_root.is_dir():
        return []
    found: list[Path] = []
    for doc_dir in sorted(p for p in docs_root.iterdir() if p.is_dir()):
        if (doc_dir / "hypotheses.json").exists():
            found.append(doc_dir)
    return found


def _load_patches(
    reviews: dict[str, list[ReviewPatch | dict[str, Any]]] | None,
    doc_id: str,
) -> list[ReviewPatch]:
    if not reviews:
        return []
    raw = reviews.get(doc_id) or reviews.get("*") or []
    out: list[ReviewPatch] = []
    for item in raw:
        if isinstance(item, ReviewPatch):
            out.append(item)
        else:
            out.append(ReviewPatch.model_validate(item))
    return out


def process_from_block4(
    input_source: str | Path,
    *,
    output_dir: str | Path | None = None,
    output_zip: str | Path | None = None,
    kg: KnowledgeGraph | None = None,
    reviews: dict[str, list[ReviewPatch | dict[str, Any]]] | None = None,
    enable_llm: bool = False,
    llm: LlmRanker | None = None,
) -> dict[str, Any]:
    """Ingest a Block 4 ZIP/folder, queue HiTL, apply patches, emit order.json.

    Leaves hypotheses.json and the original prediction.json unchanged.
    LLM suggestions (if enabled) are attached to the queue only — never auto-applied.
    """
    kg = kg or KnowledgeGraph.load(DEFAULT_KG)
    base_in_dir, is_temp_in = unzip_or_dir(input_source)
    print(f"[Block 5] Ingested {input_source}")

    if output_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix="block5_out_"))
    else:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    docs_out: list[dict[str, Any]] = []
    for doc_dir in _iter_docs(base_in_dir):
        hyp = DocumentHypotheses.model_validate_json(
            (doc_dir / "hypotheses.json").read_text(encoding="utf-8")
        )
        pred_path = doc_dir / "prediction.json"
        if pred_path.exists():
            pred = DocumentPrediction.model_validate_json(pred_path.read_text(encoding="utf-8"))
        else:
            pred = rescore_hypotheses(hyp, kg)

        print(f"  → reviewing '{hyp.doc_id}'...")
        queue = build_hitl_queue(pred, hyp, doc_dir=doc_dir)
        queue = attach_llm_suggestions(queue, ranker=llm, enabled=enable_llm, kg=kg)
        patches = _load_patches(reviews, hyp.doc_id)
        patched, committed = commit_hypotheses(hyp, patches, kg)
        _ = patched
        order = order_from_prediction(committed)
        review = DocumentReview(
            doc_id=hyp.doc_id,
            patches=patches,
            queue=queue,
            auto_passed=not queue and not patches,
        )

        doc_dir_out = target_dir / "docs" / hyp.doc_id
        doc_dir_out.mkdir(parents=True, exist_ok=True)
        for src in doc_dir.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(doc_dir)
            if rel.name in {"order.json", "review.json", "prediction.committed.json"}:
                continue
            dest = doc_dir_out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

        (doc_dir_out / "review.json").write_text(review.model_dump_json(indent=2), encoding="utf-8")
        (doc_dir_out / "prediction.committed.json").write_text(
            committed.model_dump_json(indent=2), encoding="utf-8"
        )
        (doc_dir_out / "order.json").write_text(order.model_dump_json(indent=2), encoding="utf-8")

        docs_out.append(
            {
                "doc_id": committed.doc_id,
                "is_valid": committed.is_valid,
                "needs_review": order.needs_review,
                "auto_passed": review.auto_passed,
                "n_queue": len(queue),
                "n_patches": len(patches),
                "n_ticked": len(committed.ticked_test_ids),
                "n_hitl": len(committed.hitl_fields),
                "hitl_fields": committed.hitl_fields,
                "ticked_test_ids": committed.ticked_test_ids,
                "ordered_tests": order.ordered_tests,
                "observed_tubes": committed.observed_tubes,
                "expected_tubes": committed.expected_tubes,
                "review_path": f"docs/{committed.doc_id}/review.json",
                "order_path": f"docs/{committed.doc_id}/order.json",
                "committed_path": f"docs/{committed.doc_id}/prediction.committed.json",
            }
        )

    manifest = BatchPredictionManifest(
        block="block5",
        stage="review_commit",
        total_documents=len(docs_out),
        valid_documents=sum(1 for d in docs_out if d["is_valid"] and not d["needs_review"]),
        hitl_documents=sum(1 for d in docs_out if d["needs_review"]),
        documents=docs_out,
    )
    (target_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    in_manifest = base_in_dir / "manifest.json"
    if in_manifest.exists():
        shutil.copy2(in_manifest, target_dir / "block4_manifest.json")

    zip_path: Path | None = None
    if output_zip is not None:
        zip_path = Path(output_zip)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Block 5] Packaging '{zip_path}'...")
        import zipfile

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in target_dir.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, arcname=file_path.relative_to(target_dir))
        print(f"✓ Block 5 ZIP created: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")

    cleanup_temp(base_in_dir, is_temp_in)
    return {
        "manifest": manifest.model_dump(),
        "output_dir": str(target_dir),
        "output_zip": str(zip_path) if zip_path else None,
    }

"""End-to-end Blocks 1–5 on a folder, ZIP, or list of images."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from med_doc.htr.batch import process_from_block1
from med_doc.kg.graph import KnowledgeGraph
from med_doc.normalization.batch import normalize_batch
from med_doc.paths import DEFAULT_KG
from med_doc.rescoring.batch import process_from_block3
from med_doc.review.batch import process_from_block4
from med_doc.review.schemas import ReviewPatch


def run_blocks_1_to_5(
    inputs: str | Path | list,
    *,
    output_dir: str | Path | None = None,
    kg: KnowledgeGraph | None = None,
    backend: str = "lexicon",
    reviews: dict[str, list[ReviewPatch | dict[str, Any]]] | None = None,
    patch_missing_edta: bool = False,
) -> dict[str, Any]:
    """Normalize many photos, then run Blocks 3–5 on the whole batch.

    ``inputs`` is the same as Block 1a / ``normalize_batch``: a directory, a ZIP
    of images, one file, or a sequence of paths.
    """
    kg = kg or KnowledgeGraph.load(DEFAULT_KG)
    if output_dir is None:
        target = Path(tempfile.mkdtemp(prefix="pipeline_1_5_"))
    else:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

    print("[Pipeline] Block 1 — normalize batch")
    b1 = normalize_batch(
        inputs,
        output_dir=target / "b1",
        output_zip=target / "block1.zip",
    )
    print("[Pipeline] Block 3 — drafts")
    b3 = process_from_block1(
        b1["output_zip"],
        output_dir=target / "b3",
        output_zip=target / "block3.zip",
        kg=kg,
        backend=backend,
        mode="both",
    )
    print("[Pipeline] Block 4 — KG rescoring")
    b4 = process_from_block3(
        b3["output_zip"],
        output_dir=target / "b4",
        output_zip=target / "block4.zip",
        kg=kg,
    )

    applied = dict(reviews or {})
    if patch_missing_edta:
        for row in b4["manifest"]["documents"]:
            obs = row.get("observed_tubes") or {}
            if obs.get("EDTA") is None and row.get("doc_id") not in applied:
                applied[str(row["doc_id"])] = [
                    ReviewPatch(field_id="tube_edta", action="set_tube", value="1")
                ]

    print("[Pipeline] Block 5 — review / LIS")
    b5 = process_from_block4(
        b4["output_zip"],
        output_dir=target / "b5",
        output_zip=target / "block5.zip",
        kg=kg,
        reviews=applied or None,
    )
    return {
        "output_dir": str(target),
        "block1": b1,
        "block3": b3,
        "block4": b4,
        "block5": b5,
        "zips": {
            "block1": b1["output_zip"],
            "block3": b3["output_zip"],
            "block4": b4["output_zip"],
            "block5": b5["output_zip"],
        },
    }

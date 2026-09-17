"""Experimental if1 pipeline: if1block1 → 3 → 4 → original Block 5 (high-only LIS)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from med_doc.if1.if1block1 import normalize_batch as if1block1_batch
from med_doc.if1.if1block3 import process_from_if1block1
from med_doc.if1.if1block4 import process_from_if1block3
from med_doc.kg.graph import KnowledgeGraph
from med_doc.paths import DEFAULT_KG
from med_doc.review.batch import process_from_block4
from med_doc.review.schemas import OutputMode


def run_if1(
    inputs: str | Path | list,
    *,
    output_dir: str | Path | None = None,
    kg: KnowledgeGraph | None = None,
    output_mode: OutputMode = "user",
) -> dict[str, Any]:
    """Parallel to ``run_blocks_1_to_5``. Block 5 maps high-conf ticks only."""
    kg = kg or KnowledgeGraph.load(DEFAULT_KG)
    if output_dir is None:
        target = Path(tempfile.mkdtemp(prefix="if1_pipeline_"))
    else:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

    user_mode = output_mode == "user"
    work = Path(tempfile.mkdtemp(prefix="if1_work_")) if user_mode else target

    print("[if1] if1block1 — layout warp + 1b/1c")
    b1 = if1block1_batch(
        inputs,
        output_dir=work / "b1",
        output_zip=None if user_mode else target / "if1block1.zip",
    )
    print("[if1] if1block3 — geometry ticks")
    b3 = process_from_if1block1(
        b1["output_zip"] or (work / "b1"),
        output_dir=work / "b3",
        output_zip=None if user_mode else target / "if1block3.zip",
    )
    print("[if1] if1block4 — coverage tiers")
    b4 = process_from_if1block3(
        b3["output_zip"] or (work / "b3"),
        output_dir=work / "b4",
        output_zip=None if user_mode else target / "if1block4.zip",
        kg=kg,
    )
    print("[if1] Block 5 — LIS (high-confidence ticks only)")
    b5_dir = target if user_mode else (target / "b5")
    b5 = process_from_block4(
        b4["output_zip"] or (work / "b4"),
        output_dir=b5_dir,
        output_zip=None if user_mode else target / "block5.zip",
        kg=kg,
        output_mode=output_mode,
    )
    if user_mode:
        shutil.rmtree(work, ignore_errors=True)
    return {
        "output_dir": str(target),
        "output_mode": output_mode,
        "output_json": b5.get("output_json"),
        "if1block1": b1,
        "if1block3": b3,
        "if1block4": b4,
        "block5": b5,
    }

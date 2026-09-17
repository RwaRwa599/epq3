"""End-to-end Blocks 1–5 on a folder, ZIP, or list of images."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from med_doc.htr.batch import process_from_block1
from med_doc.kg.graph import KnowledgeGraph
from med_doc.normalization.batch import normalize_batch
from med_doc.paths import DEFAULT_KG
from med_doc.rescoring.batch import process_from_block3
from med_doc.review.batch import process_from_block4
from med_doc.review.schemas import OutputMode, ReviewPatch


def run_blocks_1_to_5(
    inputs: str | Path | list,
    *,
    output_dir: str | Path | None = None,
    kg: KnowledgeGraph | None = None,
    backend: str = "lexicon",
    reviews: dict[str, list[ReviewPatch | dict[str, Any]]] | None = None,
    patch_missing_edta: bool = False,
    output_mode: OutputMode = "user",
    mark_backend: str = "geometry",
    vision_backend: str = "off",
    vision_model: str | None = None,
    llm_backend: str = "off",
    llm_model: str | None = None,
    combo_backend: str = "off",
    combo_model: str | None = None,
    htr_mode: str = "both",
    if1: bool = False,
) -> dict[str, Any]:
    """Normalize many photos, then run Blocks 3–5 on the whole batch.

    ``inputs`` is the same as Block 1a / ``normalize_batch``: a directory, a ZIP
    of images, one file, or a sequence of paths.

    ``output_mode="user"`` (default): only Block 5's ``order.json`` is kept.
    ``output_mode="dev"``: Block 1/3/4/5 ZIPs plus per-document debug trees.
    ``vision_model`` is the 3c VL model (pixels). ``llm_model`` is the Block 5
    Instruct ranker (n-best strings only). ``combo_backend="ollama"`` or ``"kg"``
    is a Block 4 combination-flag placeholder. ``htr_mode="nonverbal"`` skips 3b
    handwriting. All optional backends default off.

    ``if1=True`` keeps this runner: if1block1 bar+gutter warp, stamped
    ``initial_ticked_test_ids``, if1block4 coverage high/low (then original
    Block 5). Default ``False`` is unchanged Blocks 1–5.
    """
    kg = kg or KnowledgeGraph.load(DEFAULT_KG)
    if output_dir is None:
        target = Path(tempfile.mkdtemp(prefix="pipeline_1_5_"))
    else:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

    user_mode = output_mode == "user"
    work = Path(tempfile.mkdtemp(prefix="pipeline_work_")) if user_mode else target

    b1_zip = None if user_mode else target / "block1.zip"
    b3_zip = None if user_mode else target / "block3.zip"
    b4_zip = None if user_mode else target / "block4.zip"

    if if1:
        from med_doc.if1.if1block1 import normalize_batch as if1block1_batch
        from med_doc.if1.if1block3 import process_from_if1block1
        from med_doc.if1.if1block4 import process_from_if1block3

        print("[Pipeline] if1block1 — layout warp + 1b/1c")
        b1 = if1block1_batch(
            inputs,
            output_dir=work / "b1",
            output_zip=b1_zip,
        )
        print("[Pipeline] if1block3 — geometry ticks")
        b3 = process_from_if1block1(
            b1["output_zip"] or (work / "b1"),
            output_dir=work / "b3",
            output_zip=b3_zip,
            kg=kg,
            backend=backend,
            mode=htr_mode,  # type: ignore[arg-type]
            mark_backend=mark_backend,  # type: ignore[arg-type]
            vision_backend=vision_backend,
            vision_model=vision_model,
        )
        print("[Pipeline] if1block4 — coverage tiers + KG")
        b4 = process_from_if1block3(
            b3["output_zip"] or (work / "b3"),
            output_dir=work / "b4",
            output_zip=b4_zip,
            kg=kg,
            combo_backend=combo_backend,
            combo_model=combo_model,
        )
    else:
        print("[Pipeline] Block 1 — normalize batch")
        b1 = normalize_batch(
            inputs,
            output_dir=work / "b1",
            output_zip=b1_zip,
        )
        print("[Pipeline] Block 3 — drafts")
        b3 = process_from_block1(
            b1["output_zip"] or (work / "b1"),
            output_dir=work / "b3",
            output_zip=b3_zip,
            kg=kg,
            backend=backend,
            mode=htr_mode,  # type: ignore[arg-type]
            mark_backend=mark_backend,  # type: ignore[arg-type]
            vision_backend=vision_backend,
            vision_model=vision_model,
        )
        print("[Pipeline] Block 4 — KG rescoring")
        b4 = process_from_block3(
            b3["output_zip"] or (work / "b3"),
            output_dir=work / "b4",
            output_zip=b4_zip,
            kg=kg,
            combo_backend=combo_backend,
            combo_model=combo_model,
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
    b5_dir = target if user_mode else (target / "b5")
    b5 = process_from_block4(
        b4["output_zip"] or (work / "b4"),
        output_dir=b5_dir,
        output_zip=None if user_mode else target / "block5.zip",
        kg=kg,
        reviews=applied or None,
        output_mode=output_mode,
        llm_backend=llm_backend,
        llm_model=llm_model,
    )

    if user_mode:
        shutil.rmtree(work, ignore_errors=True)

    zips: dict[str, Any] = {}
    if not user_mode:
        zips = {
            "block1": b1["output_zip"],
            "block3": b3["output_zip"],
            "block4": b4["output_zip"],
            "block5": b5["output_zip"],
        }
        if if1:
            for live_name, if1_name in (
                ("block1.zip", "if1block1.zip"),
                ("block3.zip", "if1block3.zip"),
                ("block4.zip", "if1block4.zip"),
            ):
                src = target / live_name
                if src.is_file():
                    dest = target / if1_name
                    shutil.copy2(src, dest)
                    zips[if1_name.replace(".zip", "")] = str(dest)

    out: dict[str, Any] = {
        "output_dir": str(target),
        "output_mode": output_mode,
        "output_json": b5.get("output_json"),
        "if1": if1,
        "block1": b1,
        "block3": b3,
        "block4": b4,
        "block5": b5,
        "zips": zips,
    }
    if if1:
        out["if1block1"] = b1
        out["if1block3"] = b3
        out["if1block4"] = b4
    return out

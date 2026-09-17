"""if1block3: geometry ticks; snapshot ``initial_ticked_test_ids``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from med_doc.htr.batch import process_from_block1
from med_doc.htr.schemas import DocumentHypotheses, DocumentPrediction
from med_doc.rescoring.ticks import trusted_tick_ids


def _stamp_initial(doc_dir: Path) -> None:
    hyp_path = doc_dir / "hypotheses.json"
    if not hyp_path.is_file():
        return
    hyp = DocumentHypotheses.model_validate_json(hyp_path.read_text(encoding="utf-8"))
    initial = trusted_tick_ids(hyp.nonverbal)
    hyp = hyp.model_copy(
        update={
            "block": "if1block3",
            "initial_ticked_test_ids": initial,
            "ticked_test_ids": list(initial),
        }
    )
    hyp_path.write_text(hyp.model_dump_json(indent=2), encoding="utf-8")
    pred_path = doc_dir / "prediction.json"
    if pred_path.is_file():
        pred = DocumentPrediction.model_validate_json(pred_path.read_text(encoding="utf-8"))
        pred = pred.model_copy(
            update={
                "initial_ticked_test_ids": initial,
                "ticked_test_ids": list(initial),
            }
        )
        pred_path.write_text(pred.model_dump_json(indent=2), encoding="utf-8")


def process_from_if1block1(
    input_source: str | Path,
    *,
    output_dir: str | Path | None = None,
    output_zip: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Accepts a Block 1 or if1block1 ZIP. 3c stays off unless kwargs pass it."""
    kwargs.setdefault("mark_backend", "geometry")
    kwargs.setdefault("mode", "nonverbal")
    kwargs.setdefault("vision_backend", "off")
    out = process_from_block1(
        input_source,
        output_dir=output_dir,
        output_zip=output_zip,
        **kwargs,
    )
    target = Path(out["output_dir"])
    for doc_dir in sorted((target / "docs").glob("*")):
        if doc_dir.is_dir():
            _stamp_initial(doc_dir)
    manifest_path = target / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["block"] = "if1block3"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        out["manifest"] = manifest
    if out.get("output_zip"):
        import zipfile

        zip_path = Path(out["output_zip"])
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in target.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, arcname=file_path.relative_to(target))
    return out

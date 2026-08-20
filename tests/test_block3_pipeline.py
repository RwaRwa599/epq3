"""End-to-end Block 1 ZIP -> Block 3 verbal/nonverbal hypotheses."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

from med_doc.htr.batch import process_from_block1
from med_doc.normalization.batch import normalize_batch
from med_doc.paths import SYNTHETIC_DIR


def test_block1_to_block3_zip_pipeline():
    samples = list(SYNTHETIC_DIR.glob("*.png"))
    assert len(samples) > 0, "No synthetic samples found for testing"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        block1_zip = tmp_path / "block1_normalized_batch.zip"
        block3_zip = tmp_path / "block3_predictions_batch.zip"

        b1 = normalize_batch(samples, output_zip=block1_zip)
        assert block1_zip.exists()
        assert b1["manifest"]["successful_documents"] == len(samples)

        with zipfile.ZipFile(block1_zip, "r") as zf:
            names = zf.namelist()
            assert any("crops/checkboxes" in n for n in names)
            assert any("crops/handwriting" in n for n in names)

        b3 = process_from_block1(block1_zip, output_zip=block3_zip, backend="lexicon", mode="both")
        assert block3_zip.exists()
        assert b3["manifest"]["total_documents"] == len(samples)
        assert b3["manifest"]["block"] == "block3"

        with zipfile.ZipFile(block3_zip, "r") as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert any(n.endswith("hypotheses.json") for n in names)
            assert any(n.endswith("prediction.json") for n in names)

            manifest = json.loads(zf.read("manifest.json"))
            doc_id = manifest["documents"][0]["doc_id"]
            hyp = json.loads(zf.read(f"docs/{doc_id}/hypotheses.json"))
            assert "nonverbal" in hyp
            assert "verbal" in hyp
            assert len(hyp["nonverbal"]) >= 100
            assert "others" in hyp["verbal"]
            assert "tube_edta" in hyp["verbal"]

"""End-to-end integration tests for Block 1 -> Block 2 ZIP batch processing pipeline."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
import pytest

from med_doc.kg.batch import process_batch_from_block1
from med_doc.normalization.batch import normalize_batch
from med_doc.paths import SYNTHETIC_DIR


def test_batch_normalization_and_block2_zip_pipeline():
    """Test full cycle: raw images -> Block 1 ZIP -> Block 2 Validation -> Block 3 ZIP."""
    samples = list(SYNTHETIC_DIR.glob("*.png"))
    assert len(samples) > 0, "No synthetic samples found for testing"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        block1_zip = tmp_path / "block1_normalized_batch.zip"
        block2_zip = tmp_path / "block2_validated_batch.zip"

        # 1. Run Block 1 Batch Normalization & ZIP export
        b1_res = normalize_batch(
            samples,
            output_zip=block1_zip,
        )

        assert block1_zip.exists()
        assert block1_zip.stat().st_size > 0
        assert b1_res["manifest"]["total_documents"] == len(samples)
        assert b1_res["manifest"]["successful_documents"] == len(samples)

        # Inspect Block 1 ZIP structure
        with zipfile.ZipFile(block1_zip, "r") as zf:
            namelist = zf.namelist()
            assert "manifest.json" in namelist
            assert any(name.endswith("canonical.png") for name in namelist)
            assert any(name.endswith("metadata.json") for name in namelist)
            assert any("crops/checkboxes" in name for name in namelist)
            assert any("crops/handwriting" in name for name in namelist)

        # 2. Ingest Block 1 ZIP into Block 2, Validate & Export Block 3 ZIP
        b2_res = process_batch_from_block1(
            input_source=block1_zip,
            output_zip=block2_zip,
        )

        assert block2_zip.exists()
        assert block2_zip.stat().st_size > 0
        assert b2_res["manifest"]["total_documents"] == len(samples)

        # Inspect Block 2 / Block 3 ZIP structure
        with zipfile.ZipFile(block2_zip, "r") as zf:
            namelist = zf.namelist()
            assert "manifest.json" in namelist
            assert any(name.endswith("validation_report.json") for name in namelist)
            assert any(name.endswith("prior_rankings.json") for name in namelist)
            assert any(name.endswith("canonical.png") for name in namelist)
            assert any("crops/handwriting" in name for name in namelist)

            manifest_content = json.loads(zf.read("manifest.json"))
            assert manifest_content["block"] == "block2"
            assert manifest_content["total_documents"] == len(samples)
            doc_entry = manifest_content["documents"][0]
            assert "expected_tubes" in doc_entry
            assert "ticked_tests" in doc_entry

"""End-to-end Blocks 1 → 3 → 4 → 5 on synthetic sheets (no PHI)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2

from med_doc.htr.batch import process_from_block1
from med_doc.kg import KnowledgeGraph
from med_doc.normalization.batch import normalize_batch
from med_doc.paths import DEFAULT_KG, SYNTHETIC_DIR
from med_doc.rescoring import process_from_block3
from med_doc.review import ReviewPatch, process_from_block4
from med_doc.template import load_template
from test_block3_verbal_nonverbal import _write_mini_block1
from test_normalization import render_canonical_form


def _run_blocks(b1_zip: Path, out: Path, *, reviews=None, backend: str = "lexicon"):
    kg = KnowledgeGraph.load(DEFAULT_KG)
    b3 = process_from_block1(
        b1_zip,
        output_zip=out / "block3.zip",
        output_dir=out / "b3",
        kg=kg,
        backend=backend,
        mode="both",
    )
    b4 = process_from_block3(
        out / "block3.zip",
        output_zip=out / "block4.zip",
        output_dir=out / "b4",
        kg=kg,
    )
    b5 = process_from_block4(
        out / "block4.zip",
        output_zip=out / "block5.zip",
        output_dir=out / "b5",
        kg=kg,
        reviews=reviews,
    )
    return b3, b4, b5


def test_mini_block1_through_block5(tmp_path: Path):
    b1 = _write_mini_block1(tmp_path / "b1")
    reviews = {"sample_sheet_01": [ReviewPatch(field_id="tube_edta", action="set_tube", value="1")]}
    b3, b4, b5 = _run_blocks(b1, tmp_path / "pipe", reviews=reviews)

    assert b3["manifest"]["block"] == "block3"
    assert b4["manifest"]["block"] == "block4"
    assert b5["manifest"]["block"] == "block5"

    with zipfile.ZipFile(tmp_path / "pipe" / "block5.zip") as zf:
        names = zf.namelist()
        assert "docs/sample_sheet_01/hypotheses.json" in names
        assert "docs/sample_sheet_01/prediction.json" in names
        assert "docs/sample_sheet_01/order.json" in names
        hyp = json.loads(zf.read("docs/sample_sheet_01/hypotheses.json"))
        order = json.loads(zf.read("docs/sample_sheet_01/order.json"))
        pred = json.loads(zf.read("docs/sample_sheet_01/prediction.json"))
        committed = json.loads(zf.read("docs/sample_sheet_01/prediction.committed.json"))

    assert hyp["nonverbal"]["cbc"]["is_marked"] is True
    assert hyp["verbal"]["tube_edta"]["source"] != "prior_expected"
    assert hyp["verbal"]["tube_edta"]["raw_text"] == ""
    assert pred["observed_tubes"].get("EDTA") is None
    assert committed["observed_tubes"].get("EDTA") == 1
    assert order["observed_tubes"]["EDTA"] == 1
    assert "cbc" in order["ticked_test_ids"]
    assert order["needs_review"] is False


def test_rendered_form_blocks_1_to_5(tmp_path: Path):
    template = load_template()
    page = render_canonical_form(template)
    cbc = template.field_map()["cbc"]
    x0, y0, x1, y1 = template.pixel_bbox(cbc, apply_pad=False)
    cv2.line(page, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (20, 20, 20), 2)
    tube = template.field_map()["tube_edta"]
    tx0, ty0, tx1, ty1 = template.pixel_bbox(tube, apply_pad=False)
    cv2.putText(
        page,
        "1",
        (tx0 + 4, ty1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    blank = SYNTHETIC_DIR / "lab_request_v0_blank.png"
    if blank.exists():
        # Prefer the checked-in blank canvas when present; overlay the same marks.
        loaded = cv2.imread(str(blank))
        if loaded is not None:
            page = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
            cv2.line(page, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (20, 20, 20), 2)
            cv2.putText(
                page,
                "1",
                (tx0 + 4, ty1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (25, 25, 25),
                2,
                cv2.LINE_AA,
            )

    sheet = tmp_path / "sheet.png"
    cv2.imwrite(str(sheet), cv2.cvtColor(page, cv2.COLOR_RGB2BGR))
    b1_zip = tmp_path / "block1.zip"
    b1 = normalize_batch([sheet], output_zip=b1_zip)
    assert b1["manifest"]["successful_documents"] == 1

    b3, b4, b5 = _run_blocks(b1_zip, tmp_path / "full", backend="lexicon")
    assert b5["manifest"]["total_documents"] == 1
    doc = b5["manifest"]["documents"][0]
    order = json.loads((tmp_path / "full" / "b5" / "docs" / doc["doc_id"] / "order.json").read_text())
    hyp = json.loads((tmp_path / "full" / "b3" / "docs" / doc["doc_id"] / "hypotheses.json").read_text())
    assert "cbc" in hyp["nonverbal"]
    assert "tube_edta" in hyp["verbal"]
    assert hyp["verbal"]["tube_edta"]["source"] != "prior_expected"
    assert isinstance(order["ordered_tests"], list)
    assert "needs_review" in order


def test_run_blocks_1_to_5_batch_folder(tmp_path: Path):
    from med_doc.pipeline import run_blocks_1_to_5

    blank = SYNTHETIC_DIR / "lab_request_v0_blank.png"
    assert blank.exists()
    folder = tmp_path / "photos"
    folder.mkdir()
    import shutil

    shutil.copy2(blank, folder / "form_a.png")
    shutil.copy2(blank, folder / "form_b.png")
    pipe = run_blocks_1_to_5(folder, output_dir=tmp_path / "out", backend="lexicon")
    assert pipe["block1"]["manifest"]["successful_documents"] == 2
    assert pipe["block3"]["manifest"]["total_documents"] == 2
    assert pipe["block4"]["manifest"]["total_documents"] == 2
    assert pipe["block5"]["manifest"]["total_documents"] == 2
    ids = {d["doc_id"] for d in pipe["block5"]["manifest"]["documents"]}
    assert ids == {"form_a", "form_b"}
    for doc_id in ids:
        hyp = json.loads((tmp_path / "out" / "b3" / "docs" / doc_id / "hypotheses.json").read_text())
        assert hyp["verbal"]["tube_edta"]["source"] != "prior_expected"
        order = json.loads((tmp_path / "out" / "b5" / "docs" / doc_id / "order.json").read_text())
        assert "needs_review" in order

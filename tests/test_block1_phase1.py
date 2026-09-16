"""Phase 1: handwriting bboxes, 1c type gates, page alignment, no silent skip."""

from __future__ import annotations

import cv2
import numpy as np

from med_doc.normalization.block1a import Block1aPage
from med_doc.normalization.block1b import Block1bLayout
from med_doc.normalization.block1c import (
    handwriting_window_ok,
    run_block1c,
    text_box_ok,
    tube_window_ok,
)
from med_doc.normalization.crops import dump_handwriting_crops, recrop_pixels
from med_doc.normalization.gates import (
    MIN_ALIGNMENT_CONFIDENCE,
    alignment_gate,
    batch_template_conflict,
    template_pick_meta,
)
from med_doc.paths import V1_TEMPLATE
from med_doc.schemas import FieldSpec, NormalizedDocumentResult, TemplateSpec
from med_doc.template import load_template


def _boxes_overlap(a: list[float], b: list[float], gap: float = 0.0) -> bool:
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def test_v1_tubes_sit_below_cardiovascular_labels():
    template = load_template(V1_TEMPLATE)
    by_id = template.field_map()
    for label_id in ("nt_probnp", "lipoprotein_a"):
        for tube_id in template.handwriting_order:
            if not str(tube_id).startswith("tube_"):
                continue
            assert not _boxes_overlap(by_id[label_id].bbox, by_id[tube_id].bbox), (
                tube_id,
                by_id[tube_id].bbox,
                label_id,
                by_id[label_id].bbox,
            )
    lip_y1 = by_id["lipoprotein_a"].bbox[3]
    assert by_id["tube_edta"].bbox[1] >= lip_y1 + 0.04
    assert by_id["clinical_info"].bbox[1] >= 0.08
    assert by_id["clinical_info"].bbox[3] <= by_id["body_check_plan_1"].bbox[1] + 0.002
    assert by_id["office_other"].bbox[1] >= by_id["tube_edta"].bbox[3]
    assert len(template.handwriting_fields()) == 12


def _form_with_labels(template) -> np.ndarray:
    w, h = template.width, template.height
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (int(0.02 * w), int(0.02 * h)), (int(0.98 * w), int(0.07 * h)), (30, 30, 30), -1)
    cv2.line(img, (int(0.03 * w), int(0.945 * h)), (int(0.97 * w), int(0.945 * h)), (40, 40, 40), 3)
    for spec in template.checkbox_fields():
        x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
        cv2.rectangle(img, (x0, y0), (x1, y1), (150, 150, 150), 2)
    by_id = template.field_map()
    for fid, text in (("nt_probnp", "NT-proBNP"), ("lipoprotein_a", "Lipoprotein (a)")):
        spec = by_id[fid]
        x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
        cv2.putText(img, text, (x1 + 4, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    for spec in template.handwriting_fields():
        x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
        if spec.field_id.startswith("tube_"):
            cv2.line(img, (x0, y1 - 3), (x1, y1 - 3), (60, 60, 60), 2)
            cv2.putText(
                img,
                spec.label.replace(" x", ""),
                (max(0, x0 - 50), y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (40, 40, 40),
                1,
            )
        else:
            cv2.rectangle(img, (x0, y0), (x1, y1), (180, 180, 180), 1)
    return img


def test_dump_v1_handwriting_crops(tmp_path):
    template = load_template(V1_TEMPLATE)
    page = _form_with_labels(template)
    out = tmp_path / "handwriting"
    paths = dump_handwriting_crops(page, template, out)
    assert len(paths) == 12
    assert {p.stem for p in paths} == set(template.handwriting_order)
    for path in paths:
        assert path.stat().st_size > 50
    edta = cv2.cvtColor(cv2.imread(str(out / "tube_edta.png")), cv2.COLOR_BGR2RGB)
    assert tube_window_ok(edta) is True
    # A crop of the lipoprotein *label* must not pass as a tube window.
    spec = template.field_map()["lipoprotein_a"]
    x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
    label = page[y0 : y1 + 4, x1 : min(template.width, x1 + 220)]
    assert tube_window_ok(label) is False


def test_tube_window_rejects_printed_names():
    img = np.full((40, 160, 3), 245, dtype=np.uint8)
    cv2.putText(img, "NT-proBNP", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    cv2.putText(img, "Lipoprotein (a)", (4, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    assert tube_window_ok(img) is False
    tail = np.full((28, 140, 3), 245, dtype=np.uint8)
    cv2.putText(tail, "proBNP", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    assert tube_window_ok(tail) is False


def test_text_box_rejects_header_bar():
    img = np.full((80, 200, 3), 30, dtype=np.uint8)
    assert text_box_ok(img) is False
    blank = np.full((80, 200, 3), 245, dtype=np.uint8)
    cv2.rectangle(blank, (2, 2), (197, 77), (160, 160, 160), 1)
    assert text_box_ok(blank) is True


def test_block1c_validates_handwriting_and_flags_body_text():
    canvas = np.full((200, 400, 3), 245, dtype=np.uint8)
    cv2.line(canvas, (20, 170), (90, 170), (50, 50, 50), 2)
    cv2.putText(canvas, "NT-proBNP / Lp(a)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    cv2.putText(canvas, "more printed names", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    tube = recrop_pixels(canvas, "tube_edta", [18, 150, 95, 185], field_type="handwriting_line")
    bad = recrop_pixels(canvas, "tube_cb", [18, 20, 220, 70], field_type="handwriting_line")
    tube = tube.model_copy(update={"field_type": "handwriting_line"})
    bad = bad.model_copy(update={"field_type": "handwriting_line"})
    assert handwriting_window_ok(tube) is True
    assert handwriting_window_ok(bad) is False
    spec = FieldSpec(field_id="cea", field_type="checkbox", bbox=[0.4, 0.4, 0.45, 0.48])
    template = TemplateSpec(template_id="t", canvas_size=[400, 200], fields=[spec])
    cv2.rectangle(canvas, (160, 80), (178, 98), (90, 90, 90), 2)
    cb = recrop_pixels(canvas, "cea", [158, 78, 180, 100])
    result = NormalizedDocumentResult(
        document_id="t",
        canonical_canvas=canvas,
        alignment_confidence=0.75,
        checkbox_crops={"cea": cb},
        handwriting_crops={"tube_edta": tube, "tube_cb": bad},
    )
    page = Block1aPage(
        canvas=canvas,
        template=template,
        warp_meta={},
        align_meta={"confidence": 0.75},
        col_shifts={},
        document_id="t",
    )
    gated = run_block1c(page, Block1bLayout(result=result, sectioned=template), draw_debug=False)
    assert gated.result.handwriting_crops["tube_edta"].crop_ok is True
    assert gated.result.handwriting_crops["tube_cb"].crop_ok is False
    assert gated.result.handwriting_crops["tube_cb"].crop_needs_hitl is True


def test_block1c_large_cell_is_not_silent_ok():
    canvas = np.full((120, 120, 3), 245, dtype=np.uint8)
    spec = FieldSpec(field_id="cea", field_type="checkbox", bbox=[0.1, 0.1, 0.7, 0.7])
    template = TemplateSpec(template_id="t", canvas_size=[120, 120], fields=[spec])
    crop = recrop_pixels(canvas, "cea", [8, 8, 80, 80])
    assert min(crop.canonical_bbox[2] - crop.canonical_bbox[0], crop.canonical_bbox[3] - crop.canonical_bbox[1]) >= 36
    result = NormalizedDocumentResult(
        document_id="t",
        canonical_canvas=canvas,
        alignment_confidence=0.8,
        checkbox_crops={"cea": crop},
        handwriting_crops={},
    )
    page = Block1aPage(
        canvas=canvas,
        template=template,
        warp_meta={},
        align_meta={},
        col_shifts={},
        document_id="t",
    )
    gated = run_block1c(page, Block1bLayout(result=result, sectioned=template), draw_debug=False)
    cea = gated.result.checkbox_crops["cea"]
    assert cea.crop_validate_status != "skip" or cea.crop_ok is False
    assert cea.crop_ok is False
    assert cea.crop_needs_hitl is True


def test_block1c_page_align_gate_keeps_per_field_tags():
    canvas = np.full((80, 80, 3), 245, dtype=np.uint8)
    cv2.rectangle(canvas, (20, 20), (38, 38), (90, 90, 90), 2)
    spec = FieldSpec(field_id="cea", field_type="checkbox", bbox=[0.2, 0.2, 0.5, 0.5])
    template = TemplateSpec(template_id="t", canvas_size=[80, 80], fields=[spec])
    crop = recrop_pixels(canvas, "cea", [18, 18, 42, 42])
    result = NormalizedDocumentResult(
        document_id="t",
        canonical_canvas=canvas,
        alignment_confidence=0.36,
        checkbox_crops={"cea": crop},
        handwriting_crops={},
    )
    page = Block1aPage(
        canvas=canvas,
        template=template,
        warp_meta={},
        align_meta={"confidence": 0.36},
        col_shifts={},
        document_id="t",
        extra={"needs_review": True},
    )
    gated = run_block1c(page, Block1bLayout(result=result, sectioned=template), draw_debug=False)
    cea = gated.result.checkbox_crops["cea"]
    assert gated.result.extra["needs_review"] is True
    assert gated.result.extra["page_gate"]["ok"] is False
    assert gated.result.extra["crop_validate"]["page_align_fail"] is True
    assert cea.crop_ok is True
    assert cea.crop_needs_hitl is False
    assert cea.crop_validate_status == "ok"


def test_block1c_neighbour_fill_recovers_column_offset():
    canvas = np.full((200, 80, 3), 245, dtype=np.uint8)
    for y in (20, 80, 140):
        cv2.rectangle(canvas, (20, y), (38, y + 18), (90, 90, 90), 2)
    template = TemplateSpec(
        template_id="t",
        canvas_size=[80, 200],
        fields=[
            FieldSpec(field_id="a", field_type="checkbox", bbox=[0.25, 0.10, 0.48, 0.19]),
            FieldSpec(field_id="b", field_type="checkbox", bbox=[0.25, 0.40, 0.48, 0.49]),
            FieldSpec(field_id="c", field_type="checkbox", bbox=[0.25, 0.70, 0.48, 0.79]),
        ],
    )
    crops = {
        "a": recrop_pixels(canvas, "a", [18, 18, 42, 42]),
        "b": recrop_pixels(canvas, "b", [18, 78, 42, 102]),
        "c": recrop_pixels(canvas, "c", [50, 4, 74, 28]),
    }
    result = NormalizedDocumentResult(
        document_id="t",
        canonical_canvas=canvas,
        alignment_confidence=0.8,
        checkbox_crops=crops,
        handwriting_crops={},
    )
    page = Block1aPage(
        canvas=canvas, template=template, warp_meta={}, align_meta={}, col_shifts={}, document_id="t"
    )
    gated = run_block1c(page, Block1bLayout(result=result, sectioned=template), draw_debug=False)
    c = gated.result.checkbox_crops["c"]
    assert c.crop_ok is True
    cy = 0.5 * (c.canonical_bbox[1] + c.canonical_bbox[3])
    assert 130 <= cy <= 165
    assert gated.result.extra["crop_validate"]["n_neighbour_fill"] >= 1


def test_block1c_retry_does_not_mutate_template():
    canvas = np.full((160, 220, 3), 245, dtype=np.uint8)
    cv2.rectangle(canvas, (30, 50), (48, 68), (90, 90, 90), 2)
    cv2.putText(canvas, "CEA", (52, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
    spec = FieldSpec(field_id="cea", field_type="checkbox", bbox=[0.22, 0.28, 0.50, 0.48], label="CEA")
    template = TemplateSpec(template_id="t", canvas_size=[220, 160], fields=[spec])
    orig = list(template.fields[0].bbox)
    crop = recrop_pixels(canvas, "cea", [52, 48, 110, 78])
    result = NormalizedDocumentResult(
        document_id="t",
        canonical_canvas=canvas,
        alignment_confidence=0.8,
        checkbox_crops={"cea": crop},
        handwriting_crops={},
    )
    page = Block1aPage(
        canvas=canvas, template=template, warp_meta={}, align_meta={}, col_shifts={}, document_id="t"
    )
    gated = run_block1c(page, Block1bLayout(result=result, sectioned=template), draw_debug=False)
    assert list(gated.sectioned.fields[0].bbox) == orig
    assert gated.result.extra["crop_validate"]["template_persist"] is False


def test_alignment_gate_threshold():
    assert alignment_gate(0.59)["ok"] is False
    assert alignment_gate(0.60)["ok"] is True
    assert MIN_ALIGNMENT_CONFIDENCE == 0.6


def test_batch_template_conflict_flags_mixed_ids():
    mixed = batch_template_conflict(
        [
            {"status": "success", "template_id": "lab_request_canonical"},
            {"status": "success", "template_id": "lab_request_v1_canonical"},
        ]
    )
    assert mixed["conflict"] is True
    same = batch_template_conflict(
        [
            {"status": "success", "template_id": "lab_request_v1_canonical"},
            {"status": "success", "template_id": "lab_request_v1_canonical"},
        ]
    )
    assert same["conflict"] is False


def test_template_pick_relative_margin_flags_coin_flip():
    # IMG_7600: 2.5 pts on ~160 is 1.6%, not a confident pick.
    close = template_pick_meta({"score_v0": 161.0, "score_v1": 158.5, "picked": "v0"})
    assert close["ambiguous"] is True
    assert close["both_failed"] is False
    assert close["relative_margin"] < 0.10
    far = template_pick_meta({"score_v0": 10.0, "score_v1": 80.0, "picked": "v1"})
    assert far["ambiguous"] is False
    assert far["both_failed"] is False


def test_template_pick_zero_tie_is_registration_failure_not_ambiguous():
    tied = template_pick_meta({"score_v0": 0.0, "score_v1": 0.0, "picked": "v0"})
    assert tied["ambiguous"] is False
    assert tied["both_failed"] is True
    assert "both templates failed" in tied["note"]


def _square_crop(size: int = 24, *, slash: bool = False) -> np.ndarray:
    img = np.full((size, size, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (1, 1), (size - 2, size - 2), (90, 90, 90), 2)
    if slash:
        for i in range(5, size - 5):
            img[i, i] = 15
            if i + 1 < size - 5:
                img[i, i + 1] = 15
    return img


def _glyph_crop(text: str, size: int = 28) -> np.ndarray:
    img = np.full((size, size, 3), 245, dtype=np.uint8)
    cv2.putText(img, text, (1, size - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 1)
    return img


def test_has_hollow_ring_accepts_square_rejects_glyphs():
    from med_doc.normalization.block1c import has_hollow_ring

    assert has_hollow_ring(_square_crop()) is True
    assert has_hollow_ring(_square_crop(slash=True)) is True
    padded = np.full((36, 36, 3), 245, dtype=np.uint8)
    padded[6:30, 6:30] = _square_crop(24)
    assert has_hollow_ring(padded) is True
    for text in ("Body", "HbA1c", "o", "B", "R"):
        assert has_hollow_ring(_glyph_crop(text)) is False, text


def test_snap_overlay_ignores_letter_loops_to_the_right():
    from med_doc.normalization.align import snap_overlay

    template = load_template(V1_TEMPLATE)
    w, h = template.width, template.height
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    for spec in template.checkbox_fields():
        x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
        cv2.rectangle(img, (x0, y0), (x1, y1), (140, 140, 140), 2)
        cx = min(w - 10, x1 + 18)
        cy = (y0 + y1) // 2
        cv2.circle(img, (cx, cy), 16, (25, 25, 25), 2)
        cv2.putText(img, "Bo", (min(w - 40, x1 + 30), y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
    _snapped, meta = snap_overlay(img, template)
    assert abs(float(meta.get("dx") or 0.0)) < 12.0

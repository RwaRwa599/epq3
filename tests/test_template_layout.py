"""Header-row template audit + label-neighbor check (no clinic photos)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from med_doc.htr.blank import blank_canvas, render_blank_form
from med_doc.htr.verbal import recognize_verbal
from med_doc.paths import PRIVATE_SAMPLES_DIR, V1_TEMPLATE
from med_doc.schemas import FieldSpec, TemplateSpec
from med_doc.template import load_template
from med_doc.template_layout import (
    audit_header_transitions,
    audit_label_neighbors,
    header_transition_fields,
    label_neighbor_match,
    looks_like_htr_garbage,
)


def test_v1_header_transitions_not_off_by_one_row():
    template = load_template(V1_TEMPLATE)
    firsts = {h.field_id for h in header_transition_fields(template)}
    assert "creatinine" in firsts
    assert "ebv_dna" in firsts
    assert "cbc" in firsts
    issues = audit_header_transitions(template)
    assert issues == [], issues


def test_renal_urea_shares_electrolyte_row():
    template = load_template(V1_TEMPLATE)
    by = template.field_map()
    assert abs(by["urea"].bbox[1] - by["na"].bbox[1]) < 0.002
    assert abs(by["k"].bbox[1] - by["na"].bbox[1]) < 0.002
    assert by["creatinine"].bbox[3] < by["urea"].bbox[1]
    assert by["urea"].bbox[3] < by["egfr"].bbox[1]
    assert by["egfr"].bbox[3] < by["alp"].bbox[1]


def test_label_neighbor_catches_one_row_shift():
    w, h = 480, 220
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (8, 36), (200, 58), (30, 30, 30), -1)
    cv2.putText(img, "MOLECULAR", (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    cv2.rectangle(img, (20, 88), (38, 106), (90, 90, 90), 2)
    cv2.putText(img, "HBV DNA", (44, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    good = FieldSpec(
        field_id="hbv_dna",
        field_type="checkbox",
        bbox=[20 / w, 88 / h, 38 / w, 106 / h],
        label="HBV DNA (quantitative)",
        group="molecular",
    )
    bad = FieldSpec(
        field_id="hbv_dna",
        field_type="checkbox",
        bbox=[20 / w, 38 / h, 38 / w, 56 / h],
        label="HBV DNA (quantitative)",
        group="molecular",
    )
    spec = TemplateSpec(template_id="t", canvas_size=[w, h], fields=[good])
    assert label_neighbor_match(img, spec, good)["ok"] is True
    spec_bad = TemplateSpec(template_id="t", canvas_size=[w, h], fields=[bad])
    assert label_neighbor_match(img, spec_bad, bad)["ok"] is False


def test_v1_label_neighbors_on_canonical_if_present():
    png = PRIVATE_SAMPLES_DIR / "canonical.png"
    if not png.is_file():
        pytest.skip("gitignored canonical.png not present")
    bgr = cv2.imread(str(png))
    assert bgr is not None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    template = load_template(V1_TEMPLATE)
    if rgb.shape[1] != template.width or rgb.shape[0] != template.height:
        pytest.skip("canonical.png is not the v1 canvas size")
    rows = audit_label_neighbors(rgb, template)
    failed = [r["field_id"] for r in rows if not r["ok"]]
    assert failed == [], failed


def test_charset_soup_is_garbage_not_a_write_in():
    soup = "D 0\nKA K LV  T K A 0\nN Y 7 X I 7  1  LZ J44ALE / 0"
    assert looks_like_htr_garbage(soup) is True
    assert looks_like_htr_garbage("Pus swab x C/ST") is False
    img = np.full((48, 240, 3), 245, dtype=np.uint8)
    cv2.putText(img, "N H N  I N Z T", (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    pred = recognize_verbal(img, "others", backend="auto")
    assert pred.source in {"garbage", "empty", "ink-present"}
    assert pred.canonical_id is None


def test_v1_blank_canvas_is_not_the_v0_sheet():
    template = load_template(V1_TEMPLATE)
    canvas = blank_canvas(template.template_id, template.width, template.height)
    assert canvas.shape[0] == template.height
    rendered = render_blank_form(template, labels=True)
    assert rendered.shape == canvas.shape
    spec = template.field_map()["hbv_dna"]
    x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
    strip = rendered[y0:y1, x1 : x1 + 80]
    assert float((strip < 80).mean()) > 0.01
    sparse = render_blank_form(template, labels=False)
    sparse_strip = sparse[y0:y1, x1 : x1 + 80]
    assert float((sparse_strip < 80).mean()) < 0.005


def test_labeled_blank_residual_drops_printed_others_header():
    from med_doc.htr.blank import residual_against_blank

    template = load_template(V1_TEMPLATE)
    labeled = render_blank_form(template, labels=True)
    sparse = render_blank_form(template, labels=False)
    spec = template.field_map()["others"]
    x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
    crop = labeled[y0:y1, x0:x1]
    paper = sparse[y0:y1, x0:x1]
    assert float(residual_against_blank(crop, crop).mean()) < 8.0
    leftover = residual_against_blank(crop, paper)
    assert float((leftover > 40).mean()) > 0.001

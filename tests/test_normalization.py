from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from med_doc.normalization.align import fine_align, match_landmark
from med_doc.normalization.crops import extract_crops, normalize_crop_rgb, quality_metrics
from med_doc.normalization.pipeline import normalize_document
from med_doc.normalization.warp import (
    detect_document_quad,
    four_point_transform,
    order_quad,
    warp_to_canonical,
)
from med_doc.paths import CANONICAL_SIZE, PRIVATE_SAMPLES_DIR
from med_doc.template import load_template

CLINIC_BATCH = Path("/Users/renaw/NLP/data/samples/private/clinic_batch")


def _looks_like_checkbox(crop: np.ndarray) -> bool:
    """Padded hollow square: bright center (paper) and a gray ring inside the crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    if min(gray.shape) < 6:
        return False
    cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
    center = gray[max(0, cy - 1) : cy + 2, max(0, cx - 1) : cx + 2]
    dark_frac = float((gray < 190).mean())
    return float(center.mean()) > 210 and 0.04 <= dark_frac <= 0.65


def render_canonical_form(template) -> np.ndarray:
    w, h = template.width, template.height
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # Printed header bar and footer rule (landmarks).
    cv2.rectangle(img, (int(0.02 * w), int(0.02 * h)), (int(0.98 * w), int(0.07 * h)), (30, 30, 30), -1)
    cv2.line(img, (int(0.03 * w), int(0.90 * h)), (int(0.97 * w), int(0.90 * h)), (40, 40, 40), 4)
    for spec in template.checkbox_fields():
        x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
        cv2.rectangle(img, (x0, y0), (x1, y1), (150, 150, 150), 2)
    return img


def photograph(page: np.ndarray, *, tilt: float = 0.12, rotate180: bool = False) -> np.ndarray:
    """Paste a page onto a darker desk with a mild perspective tilt."""
    h, w = page.shape[:2]
    desk_h, desk_w = int(h * 1.25), int(w * 1.25)
    desk = np.full((desk_h, desk_w, 3), 90, dtype=np.uint8)
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    inset_x, inset_y = (desk_w - w) // 2, (desk_h - h) // 2
    dx = tilt * w
    dst = np.array(
        [
            [inset_x + dx, inset_y],
            [inset_x + w - dx, inset_y + 8],
            [inset_x + w, inset_y + h],
            [inset_x, inset_y + h - 12],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(page, matrix, (desk_w, desk_h), borderValue=(90, 90, 90))
    if rotate180:
        warped = cv2.rotate(warped, cv2.ROTATE_180)
    return warped


def test_section_territories_cover_checkbox_groups():
    from med_doc.paths import V1_TEMPLATE

    v0 = load_template()
    assert v0.main_sections()
    assert any(s.id == "checkup_profile" for s in v0.main_sections())
    assert any(s.id == "health_check" and s.kind == "sub" for s in v0.sections)
    assert any(s.parent_id == "clinical_chemistry" for s in v0.sub_sections())
    v1 = load_template(V1_TEMPLATE)
    assert any(s.id == "molecular" for s in v1.main_sections())
    others = next(s for s in v1.main_sections() if s.id == "others")
    assert others.bbox[1] > 0.5
    for spec in v0.sections + v1.sections:
        x0, y0, x1, y1 = spec.bbox
        assert 0.0 <= x0 < x1 <= 1.0
        assert 0.0 <= y0 < y1 <= 1.0


def test_snap_sections_keeps_count_without_bars():
    from med_doc.normalization.sections import snap_sections

    template = load_template()
    page = render_canonical_form(template)
    snapped, meta = snap_sections(page, template)
    assert len(snapped.main_sections()) == len(template.main_sections())
    assert len(snapped.sub_sections()) == len(template.sub_sections())
    assert "n_snapped" in meta


def test_snap_sections_locks_to_painted_headers():
    from med_doc.normalization.sections import snap_sections
    from med_doc.paths import V1_TEMPLATE

    template = load_template(V1_TEMPLATE)
    w, h = template.width, template.height
    page = np.full((h, w, 3), 255, dtype=np.uint8)
    by_col: dict[int, list] = {}
    for spec in template.main_sections():
        if spec.column is None:
            continue
        by_col.setdefault(int(spec.column), []).append(spec)
    painted = {}
    for mains in by_col.values():
        mains = sorted(mains, key=lambda s: s.bbox[1])
        for spec in mains:
            x0, y0, x1, y1 = [int(round(v * s)) for v, s in zip(spec.bbox, (w, h, w, h))]
            bar_y1 = min(h, y0 + 28)
            cv2.rectangle(page, (x0, y0), (x1, bar_y1), (20, 20, 20), -1)
            painted[spec.id] = y0
    snapped, meta = snap_sections(page, template)
    assert meta["n_snapped"] == len(painted)
    by_id = {s.id: s for s in snapped.main_sections()}
    for sid, y0 in painted.items():
        top = int(round(by_id[sid].bbox[1] * h))
        assert abs(top - y0) <= 3, sid
    checkup_subs = snapped.sub_sections("checkup_profile")
    assert len(checkup_subs) == 2
    bar_bottom = painted["checkup_profile"] + 28
    assert checkup_subs[0].bbox[1] * h >= bar_bottom - 4
    chem_subs = snapped.sub_sections("clinical_chemistry")
    assert len(chem_subs) == 4
    for a, b in zip(chem_subs, chem_subs[1:]):
        assert a.bbox[3] <= b.bbox[1] + 1e-6
    assert "row_counts" in meta
    # Right edges must not cover the next column's tick-box side.
    cols: dict[int, list] = {}
    for spec in snapped.main_sections():
        if spec.column is None:
            continue
        cols.setdefault(int(spec.column), []).append(spec)
    for col in sorted(cols)[:-1]:
        for left in cols[col]:
            for right in cols.get(col + 1, []):
                if left.bbox[3] <= right.bbox[1] or right.bbox[3] <= left.bbox[1]:
                    continue
                assert left.bbox[2] <= right.bbox[0] + 1e-6, (left.id, right.id)


def test_detect_printed_rows_counts_squares():
    from med_doc.normalization.sections import detect_printed_rows

    h, w = 200, 300
    page = np.full((h, w), 255, dtype=np.uint8)
    for i, y in enumerate((40, 70, 100, 130, 160)):
        cv2.rectangle(page, (20, y), (36, y + 16), (40, 40, 40), 2)
        cv2.putText(page, f"row{i}", (44, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 20, 1)
    bbox = [0.02, 0.10, 0.90, 0.95]
    ys = detect_printed_rows(page, bbox, paper=255.0, height=h, width=w, skip_top=8)
    assert len(ys) == 5


def test_rows_lock_to_first_printed_peak():
    from med_doc.normalization.sections import _rows_in_leaf
    from med_doc.schemas import FieldSpec, SectionSpec

    h, w = 200, 240
    page = np.full((h, w), 255, dtype=np.uint8)
    peaks = (50, 82, 114)
    for y in peaks:
        cv2.rectangle(page, (16, y), (32, y + 16), (40, 40, 40), 2)
        cv2.putText(page, "t", (40, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 20, 1)
    leaf = SectionSpec(
        id="demo",
        kind="main",
        label="DEMO",
        bbox=[0.02, 0.10, 0.90, 0.90],
        column=2,
        groups=["demo"],
    )
    # Field Y is biased down toward the next square, as snap_overlay does on clinic scans.
    fields = [
        FieldSpec(
            field_id=f"f{i}",
            label=f"row{i}",
            field_type="checkbox",
            bbox=[0.1, 0.28 + i * 0.16, 0.2, 0.34 + i * 0.16],
            group="demo",
        )
        for i in range(3)
    ]
    rows = _rows_in_leaf(page, leaf, leaf, fields, paper=255.0, height=h, width=w)
    assert len(rows) == 3
    assert abs(int(rows[0].bbox[1] * h) - peaks[0]) <= 2
    assert rows[0].field_id == "f0"


def test_split_row_text_leaves_tick_on_the_left():
    from med_doc.normalization.sections import split_row_text_and_ticks
    from med_doc.schemas import SectionSpec

    h, w = 40, 220
    page = np.full((h, w), 255, dtype=np.uint8)
    cv2.rectangle(page, (8, 10), (24, 26), (30, 30, 30), 2)
    cv2.putText(page, "HbA1c", (32, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 20, 1)
    row = SectionSpec(
        id="demo_row",
        kind="row",
        label="HbA1c",
        bbox=[0.0, 0.0, 1.0, 1.0],
        column=0,
        groups=["diabetes"],
        field_id="hba1c",
    )
    texts, ticks = split_row_text_and_ticks(
        page, row, paper=255.0, height=h, width=w, wall_x0=0.0
    )
    assert len(texts) == 1
    assert len(ticks) == 1
    assert ticks[0].bbox[0] == pytest.approx(0.0, abs=0.01)
    assert ticks[0].bbox[2] <= texts[0].bbox[0] + 0.02
    assert ticks[0].bbox[2] - ticks[0].bbox[0] > 0.08
    assert ticks[0].field_id == "hba1c"


def test_apply_tick_windows_snaps_square_inside_strip():
    from med_doc.normalization.sections import apply_tick_windows
    from med_doc.schemas import FieldSpec, SectionSpec, TemplateSpec

    h, w = 40, 220
    page = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(page, (8, 10), (24, 26), (30, 30, 30), 2)
    cv2.putText(page, "HbA1c", (80, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    template = TemplateSpec(
        template_id="demo_tick_windows",
        canvas_size=[w, h],
        fields=[
            FieldSpec(
                field_id="hba1c",
                field_type="checkbox",
                label="HbA1c",
                bbox=[0.40, 0.20, 0.55, 0.70],
                group="diabetes",
            )
        ],
        sections=[
            SectionSpec(
                id="demo_tick",
                kind="tick",
                label="",
                bbox=[0.0, 0.0, 0.16, 1.0],
                field_id="hba1c",
            )
        ],
    )
    gated, meta = apply_tick_windows(page, template)
    assert meta["n_applied"] == 1
    box = gated.checkbox_fields()[0].bbox
    assert box[0] < 0.12
    assert box[2] < 0.20
    assert box[2] < 0.40


def test_template_relative_coords():
    template = load_template()
    assert template.canvas_size == list(CANONICAL_SIZE)
    assert len(template.checkbox_fields()) == 124
    assert len(template.handwriting_fields()) == 9
    for spec in template.fields:
        x0, y0, x1, y1 = spec.bbox
        assert 0.0 <= x0 < x1 <= 1.0
        assert 0.0 <= y0 < y1 <= 1.0
    bbox = template.pixel_bbox(template.checkbox_fields()[0])
    assert bbox[0] >= 0 and bbox[1] >= 0
    assert bbox[2] <= template.width and bbox[3] <= template.height


def test_order_quad_corners():
    pts = np.array([[10, 80], [90, 10], [100, 90], [5, 15]], dtype=np.float32)
    ordered = order_quad(pts)
    assert ordered[0, 0] + ordered[0, 1] == pytest.approx(ordered[:, 0].min() + ordered[0, 1], abs=20)
    # TL has smallest sum, BR largest
    sums = ordered.sum(axis=1)
    assert sums[0] == sums.min()
    assert sums[2] == sums.max()


def test_homography_recovers_tilted_page():
    template = load_template()
    page = render_canonical_form(template)
    photo = photograph(page, tilt=0.10)
    canvas, meta = warp_to_canonical(photo, dest_size=tuple(template.canvas_size))
    assert canvas.shape[1] == template.width
    assert canvas.shape[0] == template.height
    assert meta["method"] in {"page-quad", "min-area-rect"}
    # Header bar should land near the top after warp+orientation.
    top = canvas[: int(template.height * 0.12)].mean()
    bottom = canvas[int(template.height * 0.85) :].mean()
    assert top < bottom


def test_orientation_recovers_upside_down():
    template = load_template()
    page = render_canonical_form(template)
    photo = photograph(page, tilt=0.08, rotate180=True)
    result = normalize_document(photo, template=template, document_id="upside")
    top = result.canonical_canvas[: int(template.height * 0.12)].mean()
    bottom = result.canonical_canvas[int(template.height * 0.85) :].mean()
    assert top < bottom
    assert result.orientation_degrees in {0, 180}


def test_clahe_flattens_gradient():
    gray = np.tile(np.linspace(40, 220, 64, dtype=np.uint8), (64, 1))
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    out = normalize_crop_rgb(rgb)
    assert out.shape == gray.shape
    assert out.std() < gray.std() * 1.5
    quality, blur, glare = quality_metrics(out)
    assert 0.0 <= quality <= 1.0
    assert 0.0 <= blur <= 1.0
    assert 0.0 <= glare <= 1.0


def test_crops_stay_inside_canvas():
    template = load_template()
    page = render_canonical_form(template)
    checkbox, handwriting = extract_crops(page, template)
    assert len(checkbox) == 124
    assert len(handwriting) == 9
    h, w = page.shape[:2]
    for crop in list(checkbox.values()) + list(handwriting.values()):
        x0, y0, x1, y1 = crop.canonical_bbox
        assert 0 <= x0 < x1 <= w
        assert 0 <= y0 < y1 <= h
        assert crop.raw_image.size > 0
        assert crop.normalized_image.size > 0
        assert 0.0 <= crop.quality_score <= 1.0


def test_checkbox_squares_centered_on_canonical():
    template = load_template()
    page = render_canonical_form(template)
    checkbox, _ = extract_crops(page, template, column_shifts={})
    centered = sum(1 for c in checkbox.values() if _looks_like_checkbox(c.raw_image))
    assert centered / len(checkbox) >= 0.95, f"centered {centered}/{len(checkbox)}"


def test_checkbox_squares_centered_after_photograph():
    template = load_template()
    page = render_canonical_form(template)
    photo = photograph(page, tilt=0.10)
    result = normalize_document(photo, template=template, document_id="tilt")
    assert result.alignment_confidence > 0.2
    assert result.debug_overlay is not None
    centered = sum(
        1 for c in result.checkbox_crops.values() if _looks_like_checkbox(c.raw_image)
    )
    n = len(result.checkbox_crops)
    assert centered / n >= 0.95, f"centered {centered}/{n} after warp"


def test_section_bar_pairing_prefers_consistent_window():
    from med_doc.normalization.align import _fit_affine_y, _pair_section_bars

    expected = [608.0, 888.0, 1032.0]
    detected = [571.0, 886.0, 1043.0, 1065.0, 1213.0]
    paired = _pair_section_bars(expected, detected)
    assert paired is not None
    exp, det = paired
    _scale, shift, residual = _fit_affine_y(exp, det)
    assert residual < 40.0
    assert abs(shift) < 80.0


def test_section_bar_affine_identity_when_aligned():
    from med_doc.normalization.align import _fit_affine_y

    expected = np.array([936.0, 1152.0, 1334.0])
    detected = expected + 1.0
    scale, shift, residual = _fit_affine_y(expected, detected)
    assert scale == pytest.approx(1.0, abs=0.02)
    assert shift == pytest.approx(1.0, abs=2.0)
    assert residual < 3.0


def test_landmark_ncc_runs():
    template = load_template()
    page = render_canonical_form(template)
    gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)
    dx, dy, score = match_landmark(gray, template.landmarks[0].bbox, mode="header_bar")
    assert score >= 0.0
    aligned, meta, shifts = fine_align(page, template)
    assert aligned.shape == page.shape
    assert "confidence" in meta
    assert isinstance(shifts, dict)


def test_detect_quad_on_desk_photo():
    template = load_template()
    page = render_canonical_form(template)
    photo = photograph(page, tilt=0.12)
    quad, meta = detect_document_quad(photo)
    assert quad.shape == (4, 2)
    warped = four_point_transform(photo, quad, dest_size=tuple(template.canvas_size))
    assert warped.shape[1] == template.width
    assert meta["method"] in {"page-quad", "min-area-rect", "full-frame"}


@pytest.mark.skipif(
    not any(CLINIC_BATCH.glob("*.png")) and not any(CLINIC_BATCH.glob("*.JPG"))
    and not any(CLINIC_BATCH.glob("*.jpg")),
    reason="private clinic photos are not present",
)
def test_clinic_photos_emit_full_crop_set():
    template = load_template()
    images = sorted(
        list(CLINIC_BATCH.glob("*.png"))
        + list(CLINIC_BATCH.glob("*.jpg"))
        + list(CLINIC_BATCH.glob("*.JPG"))
    )
    assert images, "expected clinic photos"
    # Do not copy or write PHI; only check crop cardinality and canvas size.
    img = Image.open(images[0])
    result = normalize_document(img, template=template, document_id=images[0].stem)
    assert len(result.checkbox_crops) == 124
    assert len(result.handwriting_crops) == 9
    assert result.canonical_canvas.shape[1] == template.width
    assert result.canonical_canvas.shape[0] == template.height
    assert 0.0 <= result.alignment_confidence <= 1.0
    _ = PRIVATE_SAMPLES_DIR  # documented gitignored location in this repo

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

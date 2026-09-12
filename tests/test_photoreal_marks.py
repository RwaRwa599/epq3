"""Block 3 residual imaging, logreg features, and photo-realistic isolation."""

from __future__ import annotations

import cv2
import numpy as np

from med_doc.eval.photoreal import distort_sheet, draw_ticks, shadow_gradient
from med_doc.htr.blank import residual_against_blank
from med_doc.htr.marks import classify_mark, ink_density, mark_features, train_mark_logreg
from med_doc.normalization.block1c import has_hollow_ring


def _empty(size: int = 32) -> np.ndarray:
    img = np.full((size, size, 3), 245, dtype=np.uint8)
    img[0:2, :] = 20
    img[-2:, :] = 20
    img[:, 0:2] = 20
    img[:, -2:] = 20
    return img


def _tick(size: int = 32) -> np.ndarray:
    img = _empty(size)
    for i in range(6, size - 6):
        img[i, i] = 15
        if i + 1 < size - 6:
            img[i, i + 1] = 15
    return img


def test_residual_subtracts_printed_ring():
    blank = _empty()
    tick = _tick()
    corner = np.full((32, 32, 3), 245, dtype=np.uint8)
    corner[0:4, :] = 20
    corner[:, 0:4] = 20
    r_empty = residual_against_blank(blank, blank)
    r_tick = residual_against_blank(tick, blank)
    r_corner = residual_against_blank(corner, blank)
    assert float(r_empty.mean()) < 12.0
    assert float(r_tick.mean()) > float(r_empty.mean()) + 4.0
    # Printed L-corner vs a printed square blank: residual must not look like a slash.
    pred = classify_mark(corner, "body_check_plan_1", blank=blank)
    assert pred.is_marked is False
    assert ink_density(tick, blank=blank) > ink_density(blank, blank=blank)


def test_logreg_separates_empty_and_tick():
    Xs = []
    ys = []
    for _ in range(8):
        Xs.append(mark_features(_empty()))
        ys.append(0.0)
        Xs.append(mark_features(_tick()))
        ys.append(1.0)
    w, b = train_mark_logreg(np.stack(Xs), np.asarray(ys), steps=250)
    z_empty = float(np.dot(mark_features(_empty()), w) + b)
    z_tick = float(np.dot(mark_features(_tick()), w) + b)
    assert z_tick > z_empty


def test_classification_isolated_from_page_warp():
    """Distort the crop only — registration is held fixed."""
    blank = _empty(40)
    tick = _tick(40)
    n_tp = n_fp = 0
    for seed in range(12):
        d_blank = distort_sheet(blank, seed=seed, strength=0.35)
        d_tick = distort_sheet(tick, seed=seed, strength=0.35)
        if classify_mark(d_blank, "alt").is_marked:
            n_fp += 1
        if classify_mark(d_tick, "cbc").is_marked:
            n_tp += 1
    assert n_fp <= 2
    assert n_tp >= 8


def test_registration_isolated_hollow_after_shadow():
    page = np.full((80, 80, 3), 245, dtype=np.uint8)
    cv2.rectangle(page, (20, 20), (44, 44), (110, 110, 110), 2)
    shaded = shadow_gradient(page, strength=0.6, axis=1)
    crop = shaded[18:48, 18:48]
    assert has_hollow_ring(crop) is True


def test_draw_ticks_helper_marks_boxes():
    page = np.full((60, 60, 3), 255, dtype=np.uint8)
    cv2.rectangle(page, (10, 10), (28, 28), (140, 140, 140), 2)
    ticked = draw_ticks(page, [[10, 10, 28, 28]], kind="slash")
    crop = ticked[8:32, 8:32]
    assert classify_mark(crop, "cbc").is_marked is True

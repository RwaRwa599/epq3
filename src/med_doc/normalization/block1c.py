"""Block 1c: crop-window gate after 1b. Does not classify ticks.

A checkbox PNG must be a printed square (empty hollow or ink-in-ring), never a
label, neighbour, or header. Rematch searches left of a label strip (clinic
names sit to the right of the box) and always considers ink-in-ring so a ticked
box is not skipped because the 1b window was blank paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from med_doc.normalization.block1a import Block1aPage
from med_doc.normalization.block1b import Block1bLayout
from med_doc.normalization.crops import recrop_pixels
from med_doc.normalization.gates import MIN_ALIGNMENT_CONFIDENCE, alignment_gate
from med_doc.normalization.illumination import Paper, paper_at, paper_level, paper_map
from med_doc.normalization.register import neighbour_offset, predicted_center, ransac_partial_affine
from med_doc.normalization.sections import _hollow_score, _hollow_xs, _ink_ring_candidates
from med_doc.normalization.viz import draw_overlay
from med_doc.schemas import FieldCrop, TemplateSpec

SEARCH_PAD_LEFT = 80
SEARCH_PAD_RIGHT = 48
SEARCH_PAD_Y = 36
SQUARE = 18


def _as_gray(crop: np.ndarray) -> np.ndarray:
    arr = np.asarray(crop)
    if arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def looks_like_text_line(crop: np.ndarray) -> bool:
    gray = _as_gray(crop)
    if gray.size == 0 or min(gray.shape[:2]) < 8:
        return False
    h, w = gray.shape[:2]
    paper = paper_level(gray)
    cut = paper * 0.55
    dark = gray < cut
    dens = float(dark.mean())
    if dens < 0.03:
        return False
    if w >= int(h * 1.45):
        return True
    row = dark.mean(axis=1)
    wide = row >= 0.40
    col = dark.mean(axis=0)
    return float(wide.mean()) >= 0.14 and float(col.std()) < 0.14


def has_hollow_ring(crop: np.ndarray) -> bool:
    """Printed square: dark ring with paper or ink interior. Not a header bar."""
    gray = _as_gray(crop)
    if gray.size == 0 or min(gray.shape[:2]) < 8:
        return False
    h, w = gray.shape[:2]
    if max(h, w) / float(min(h, w)) > 1.35:
        return False
    paper = paper_level(gray)
    cy, cx = h // 2, w // 2
    center = float(gray[max(0, cy - 2) : cy + 3, max(0, cx - 2) : cx + 3].mean())
    dark = float((gray < paper * 0.55).mean())
    if min(h, w) <= 24:
        band = 2
        border = np.concatenate(
            [gray[:band].ravel(), gray[-band:].ravel(), gray[:, :band].ravel(), gray[:, -band:].ravel()]
        )
        interior = gray[band:-band, band:-band]
        border_d = float((border < paper * 0.55).mean())
        inn_d = float((interior < paper * 0.55).mean()) if interior.size else 0.0
        hollow = center > paper * 0.82 and border_d >= 0.15 and inn_d <= 0.12
        ink_in = border_d >= 0.15 and inn_d >= 0.08
        if hollow or ink_in:
            return True
    ink = _ink_ring_candidates(gray, 0, h, 0, w, paper)
    xs = _hollow_xs(gray, 0, h, 0, w, paper)
    cx = w / 2.0
    near = [hx for hx, hy, _ in ink if abs(hx + 8 - cx) <= 12]
    near += [x for x in xs if abs(x + 8 - cx) <= 12]
    if near:
        return True
    y = max(0, (h - 16) // 2)
    return _hollow_score(gray, y, max(0, int(cx) - 16), min(w, int(cx) + 16), paper) >= 0.08


def window_ok(crop: np.ndarray) -> bool:
    return has_hollow_ring(crop) and not looks_like_text_line(crop)


def crop_window_ok(crop: FieldCrop) -> bool:
    """True when raw or illumination-normalized pixels look like a printed square."""
    if window_ok(crop.raw_image):
        return True
    norm = getattr(crop, "normalized_image", None)
    return norm is not None and window_ok(norm)


def _ink_density(crop: np.ndarray) -> float:
    gray = _as_gray(crop)
    if gray.size == 0:
        return 0.0
    paper = paper_level(gray)
    return float((gray < paper * 0.55).mean())


def _dark_bands(row: np.ndarray, *, on_t: float = 0.18, off_t: float = 0.08) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(row):
        if v >= on_t and start is None:
            start = i
        elif v < off_t and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, len(row)))
    return bands


def looks_like_body_text(crop: np.ndarray) -> bool:
    """Printed test names / paragraph — not a write-in blank or a single rule."""
    gray = _as_gray(crop)
    if gray.size == 0 or min(gray.shape[:2]) < 8:
        return False
    dens = _ink_density(gray)
    if dens < 0.07:
        return False
    paper = paper_level(gray)
    row = (gray < paper * 0.55).mean(axis=1)
    text_bands = [b for b in _dark_bands(row) if (b[1] - b[0]) >= 6]
    if len(text_bands) >= 3 and dens >= 0.08:
        return True
    return len(text_bands) >= 2 and dens >= 0.08 and looks_like_text_line(crop)


def has_underline(crop: np.ndarray) -> bool:
    """Thin horizontal rule — not a band of printed letters."""
    gray = _as_gray(crop)
    if gray.size == 0 or min(gray.shape[:2]) < 6:
        return False
    h, w = gray.shape[:2]
    paper = paper_level(gray)
    y0 = int(h * 0.20)
    band = gray[y0:, :]
    if band.size == 0:
        return False
    row_dark = (band < paper * 0.55).mean(axis=1)
    peak_i = int(np.argmax(row_dark))
    peak = float(row_dark[peak_i])
    n_thick = int((row_dark >= 0.22).sum())
    rule = band[peak_i]
    rule_frac = float((rule < paper * 0.55).mean())
    return peak >= 0.40 and n_thick <= 8 and rule_frac >= 0.40


def tube_window_ok(crop: np.ndarray) -> bool:
    """Short underline / blank with a printed tube label nearby — not body text."""
    gray = _as_gray(crop)
    if gray.size == 0 or min(gray.shape[:2]) < 6:
        return False
    dens = _ink_density(gray)
    if looks_like_body_text(crop):
        return False
    # Label tails (e.g. "NT-proBNP") are a text line without an underline.
    if looks_like_text_line(crop) and not has_underline(crop):
        return False
    h, w = gray.shape[:2]
    if max(h, w) / float(min(h, w)) < 1.35 and has_hollow_ring(crop):
        return False
    if dens <= 0.22:
        return True
    return bool(has_underline(crop) and dens <= 0.36)


def text_box_ok(crop: np.ndarray) -> bool:
    """Bounded blank / sparse ink — not a header bar or a stack of printed labels."""
    gray = _as_gray(crop)
    if gray.size == 0 or min(gray.shape[:2]) < 8:
        return False
    if float(gray.mean()) < 110:
        return False
    dens = _ink_density(gray)
    if dens > 0.42:
        return False
    if looks_like_body_text(crop) and dens > 0.12:
        return False
    return dens <= 0.30


def handwriting_window_ok(crop: FieldCrop) -> bool:
    img = crop.raw_image
    fid = crop.field_id
    ftype = crop.field_type
    if fid.startswith("tube_") or ftype == "handwriting_line":
        ok = tube_window_ok(img)
        if not ok and getattr(crop, "normalized_image", None) is not None:
            ok = tube_window_ok(crop.normalized_image)
        return ok
    ok = text_box_ok(img)
    if not ok and getattr(crop, "normalized_image", None) is not None:
        ok = text_box_ok(crop.normalized_image)
    return ok


def _center(bbox: list[int]) -> tuple[float, float]:
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _too_close_to_other(
    cand: tuple[float, float],
    own_id: str,
    centers: dict[str, tuple[float, float]],
    *,
    min_d: float = 20.0,
) -> bool:
    for fid, ctr in centers.items():
        if fid == own_id:
            continue
        if _dist(cand, ctr) < min_d:
            return True
    return False


def _closer_to_other(
    cand: tuple[float, float],
    own_id: str,
    centers: dict[str, tuple[float, float]],
) -> bool:
    own = centers.get(own_id)
    if own is None:
        return False
    d_own = _dist(cand, own)
    for fid, ctr in centers.items():
        if fid == own_id:
            continue
        if _dist(cand, ctr) + 1.0 < d_own:
            return True
    return False


def _dark_header(gray: np.ndarray, cx: int, cy: int, paper: Paper) -> bool:
    """Reject rematch onto a section header bar."""
    h, w = gray.shape[:2]
    y0 = max(0, cy - 22)
    y1 = max(y0 + 4, cy - 6)
    x0 = max(0, cx - 48)
    x1 = min(w, cx + 48)
    if y1 <= y0 or x1 <= x0:
        return False
    strip = gray[y0:y1, x0:x1]
    p = paper_at(paper, cy, cx)
    return float(strip.mean()) < p * 0.42 and float((strip < p * 0.40).mean()) > 0.45


def _search_candidates(
    gray: np.ndarray,
    bbox: list[int],
    paper: Paper,
    *,
    allow_ink_rings: bool,
) -> list[tuple[int, int]]:
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = bbox
    # Clinic labels sit to the right of the box; empty 1b windows sit to the left.
    sx0 = max(0, x0 - SEARCH_PAD_LEFT)
    sy0 = max(0, y0 - SEARCH_PAD_Y)
    sx1 = min(w, x1 + SEARCH_PAD_RIGHT)
    sy1 = min(h, y1 + SEARCH_PAD_Y)
    out: list[tuple[int, int]] = []
    if allow_ink_rings:
        for hx, hy, _ in _ink_ring_candidates(gray, sy0, sy1, sx0, sx1, paper):
            out.append((int(hx), int(hy)))
    for hx in _hollow_xs(gray, sy0, sy1, sx0, sx1, paper):
        # Score Y along the strip; use mid-row like _hollow_xs.
        y = max(sy0, (sy0 + sy1 - 16) // 2)
        pair = (int(hx), int(y))
        if not any(abs(pair[0] - a) < 10 and abs(pair[1] - b) < 10 for a, b in out):
            if _hollow_score(gray, y, int(hx), int(hx) + 18, paper) >= 0.08:
                out.append(pair)
    return out


def _bbox_from_xy(hx: int, hy: int, w: int, h: int) -> list[int]:
    side = SQUARE
    return [
        max(0, hx),
        max(0, hy),
        min(w, hx + side),
        min(h, hy + side),
    ]


def _tag(crop: FieldCrop, *, ok: bool, hitl: bool, status: str, attempts: int) -> FieldCrop:
    return crop.model_copy(
        update={
            "crop_ok": ok,
            "crop_needs_hitl": hitl,
            "crop_validate_status": status,
            "crop_validate_attempts": attempts,
        }
    )


@dataclass
class Block1cLayout:
    result: Any
    sectioned: TemplateSpec

    def to_result(self):
        return self.result


def run_block1c(page: Block1aPage, layout: Block1bLayout, *, draw_debug: bool = True) -> Block1cLayout:
    """Validate 1b windows. Checkboxes: printed square. Handwriting: type-specific.

    Does not classify ticks. Does not write rematches back into the shared template.
    """
    canvas = page.canvas
    result = layout.result
    template = layout.sectioned
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY) if canvas.ndim == 3 else canvas
    h, w = gray.shape[:2]
    paper = paper_map(gray, tile=64)

    extra = dict(result.extra or {})
    extra["template_pick"] = extra.get("template_pick") or page.extra.get("template_pick") or {}
    extra["n_checkbox"] = extra.get("n_checkbox") or len(page.template.checkbox_fields())
    scores = [float(result.alignment_confidence)]
    if page.align_meta.get("confidence") is not None:
        scores.append(float(page.align_meta.get("confidence") or 0.0))
    gate = alignment_gate(min(scores))
    if page.extra.get("needs_review"):
        gate = {**gate, "ok": False, "reason": gate.get("reason") or "alignment_confidence"}
    extra["page_gate"] = gate
    extra["needs_review"] = (not bool(gate["ok"])) or bool(
        (extra.get("template_pick") or {}).get("ambiguous")
    )

    crops = dict(result.checkbox_crops)
    hw = dict(result.handwriting_crops)

    n_hw_ok = n_hw_hitl = 0
    for fid, crop in list(hw.items()):
        if handwriting_window_ok(crop):
            hw[fid] = _tag(crop, ok=True, hitl=False, status="ok", attempts=0)
            n_hw_ok += 1
        else:
            hw[fid] = _tag(crop, ok=False, hitl=True, status="hitl", attempts=0)
            n_hw_hitl += 1

    centers = {fid: _center(c.canonical_bbox) for fid, c in crops.items()}
    expected: dict[str, tuple[float, float]] = {}
    for spec in template.checkbox_fields():
        x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
        expected[spec.field_id] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    n_ok = n_retry = n_hitl = n_skip = 0

    src_pts: list[list[float]] = []
    dst_pts: list[list[float]] = []
    for fid, crop in crops.items():
        if fid not in expected:
            continue
        if crop_window_ok(crop):
            src_pts.append([expected[fid][0], expected[fid][1]])
            dst_pts.append([centers[fid][0], centers[fid][1]])
    affine = None
    if len(src_pts) >= 3:
        affine, _inn = ransac_partial_affine(
            np.asarray(src_pts, dtype=np.float32),
            np.asarray(dst_pts, dtype=np.float32),
            thresh=14.0,
            min_inliers=max(3, len(src_pts) // 4),
        )

    bbox_overrides: dict[str, list[int]] = {}
    for fid, crop in list(crops.items()):
        x0, y0, x1, y1 = crop.canonical_bbox
        if crop_window_ok(crop):
            crops[fid] = _tag(crop, ok=True, hitl=False, status="ok", attempts=0)
            n_ok += 1
            continue

        candidates = _search_candidates(gray, crop.canonical_bbox, paper, allow_ink_rings=True)
        own = _center(crop.canonical_bbox)
        prior = predicted_center(affine, expected.get(fid, own))
        on_label = looks_like_text_line(crop.raw_image) or (x1 - x0) > int((y1 - y0) * 1.3)

        def _rank(p: tuple[int, int]) -> float:
            cand_c = (p[0] + 9.0, p[1] + 9.0)
            if on_label:
                return abs(cand_c[1] - own[1]) * 3.0 + cand_c[0]
            left_pen = max(0.0, cand_c[0] - own[0]) * 0.25
            return (
                0.55 * _dist(cand_c, prior)
                + 0.25 * _dist(cand_c, own)
                + left_pen
            )

        ranked = sorted(candidates, key=_rank)
        chosen: list[int] | None = None
        for hx, hy in ranked:
            cand_c = (hx + 9.0, hy + 9.0)
            if _too_close_to_other(cand_c, fid, centers):
                continue
            if _dark_header(gray, int(cand_c[0]), int(cand_c[1]), paper):
                continue
            box = _bbox_from_xy(hx, hy, w, h)
            probe = recrop_pixels(canvas, fid, box)
            if looks_like_text_line(probe.raw_image) or not has_hollow_ring(probe.raw_image):
                continue
            chosen = box
            break

        if chosen is not None:
            new_crop = recrop_pixels(canvas, fid, chosen)
            crops[fid] = _tag(new_crop, ok=True, hitl=False, status="retry", attempts=1)
            centers[fid] = _center(chosen)
            bbox_overrides[fid] = chosen
            n_retry += 1
        else:
            crops[fid] = _tag(crop, ok=False, hitl=True, status="hitl", attempts=1)
            n_hitl += 1

    cuts = None
    xs = [expected[fid][0] / float(max(w, 1)) for fid in expected]
    if len(xs) >= 4:
        cuts = np.quantile(xs, [0.25, 0.5, 0.75])

    def _col_of(xy: tuple[float, float]) -> int:
        if cuts is None:
            return 0
        return int(np.searchsorted(cuts, xy[0] / float(max(w, 1))))

    by_col: dict[int, list[str]] = {}
    for fid in expected:
        by_col.setdefault(_col_of(expected[fid]), []).append(fid)
    n_neighbour = 0
    for col, fids in by_col.items():
        src: list[list[float]] = []
        dst: list[list[float]] = []
        unresolved: list[str] = []
        for fid in fids:
            crop = crops.get(fid)
            if crop is None or fid not in expected:
                continue
            if crop.crop_ok:
                src.append([expected[fid][0], expected[fid][1]])
                dst.append([centers[fid][0], centers[fid][1]])
            elif crop.crop_needs_hitl:
                unresolved.append(fid)
        if len(src) < 2 or not unresolved:
            continue
        m, _med = neighbour_offset(
            np.asarray(src, dtype=np.float32),
            np.asarray(dst, dtype=np.float32),
        )
        for fid in unresolved:
            prior = predicted_center(m, expected[fid]) if m is not None else (
                expected[fid][0] + _med[0],
                expected[fid][1] + _med[1],
            )
            box = _bbox_from_xy(int(round(prior[0] - 9)), int(round(prior[1] - 9)), w, h)
            probe = recrop_pixels(canvas, fid, box)
            if looks_like_text_line(probe.raw_image) or not has_hollow_ring(probe.raw_image):
                continue
            if _too_close_to_other(_center(box), fid, centers):
                continue
            crops[fid] = _tag(probe, ok=True, hitl=False, status="retry", attempts=1)
            centers[fid] = _center(box)
            bbox_overrides[fid] = box
            n_retry += 1
            n_hitl = max(0, n_hitl - 1)
            n_neighbour += 1

    extra["crop_validate"] = {
        "n_ok": n_ok,
        "n_retry": n_retry,
        "n_hitl": n_hitl,
        "n_skip": n_skip,
        "n_hw_ok": n_hw_ok,
        "n_hw_hitl": n_hw_hitl,
        "n_neighbour_fill": n_neighbour,
        "ransac_inliers": len(src_pts),
        "ransac": affine is not None,
        "page_align_fail": not bool(gate["ok"]),
        "template_persist": False,
        "n_bbox_overrides": len(bbox_overrides),
        "alignment_confidence": round(float(result.alignment_confidence), 4),
        "threshold": MIN_ALIGNMENT_CONFIDENCE,
    }
    extra["bbox_overrides"] = bbox_overrides
    extra["needs_review"] = (not bool(gate["ok"])) or bool(
        (extra.get("template_pick") or {}).get("ambiguous")
    )
    result.checkbox_crops = crops
    result.handwriting_crops = hw
    result.extra = extra
    if draw_debug:
        result.debug_overlay = draw_overlay(canvas, result, template)
    return Block1cLayout(result=result, sectioned=template)

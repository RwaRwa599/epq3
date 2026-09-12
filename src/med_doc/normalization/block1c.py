"""Block 1c: crop-window gate after 1b. Does not classify ticks.

A checkbox PNG must be a printed square (empty hollow or ink-in-ring), never a
label, neighbour, or header. One widen+rematch with steal and dark-header guards;
else keep the 1b bbox and set ``crop_needs_hitl``. Overlay cells ≥36 px are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from med_doc.normalization.block1a import Block1aPage
from med_doc.normalization.block1b import Block1bLayout
from med_doc.normalization.crops import recrop_pixels
from med_doc.normalization.sections import _hollow_score, _hollow_xs, _ink_ring_candidates
from med_doc.normalization.viz import draw_overlay
from med_doc.schemas import FieldCrop, FieldSpec, TemplateSpec

OVERLAY_SKIP_PX = 36
SEARCH_PAD_PX = 28
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
    paper = float(np.percentile(gray, 90))
    cut = paper * 0.55
    dark = gray < cut
    dens = float(dark.mean())
    if dens < 0.05:
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
    paper = float(np.percentile(gray, 90))
    cy, cx = h // 2, w // 2
    center = float(gray[max(0, cy - 2) : cy + 3, max(0, cx - 2) : cx + 3].mean())
    dark = float((gray < 190).mean())
    if min(h, w) <= 24:
        band = 2
        border = np.concatenate(
            [gray[:band].ravel(), gray[-band:].ravel(), gray[:, :band].ravel(), gray[:, -band:].ravel()]
        )
        interior = gray[band:-band, band:-band]
        border_d = float((border < 190).mean())
        inn_d = float((interior < 190).mean()) if interior.size else 0.0
        hollow = center > 210 and border_d >= 0.15 and inn_d <= 0.12
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


def _interior_dark_frac(crop: np.ndarray) -> float:
    gray = _as_gray(crop)
    if gray.size == 0:
        return 0.0
    h, w = gray.shape[:2]
    inn = gray[h // 4 : max(h // 4 + 1, 3 * h // 4), w // 4 : max(w // 4 + 1, 3 * w // 4)]
    if inn.size == 0:
        return 0.0
    paper = float(np.percentile(gray, 90))
    return float((inn < paper * 0.55).mean())


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


def _dark_header(gray: np.ndarray, cx: int, cy: int, paper: float) -> bool:
    """Reject rematch onto a section header bar."""
    h, w = gray.shape[:2]
    y0 = max(0, cy - 22)
    y1 = max(y0 + 4, cy - 6)
    x0 = max(0, cx - 48)
    x1 = min(w, cx + 48)
    if y1 <= y0 or x1 <= x0:
        return False
    strip = gray[y0:y1, x0:x1]
    return float(strip.mean()) < paper * 0.42 and float((strip < paper * 0.40).mean()) > 0.45


def _search_candidates(
    gray: np.ndarray,
    bbox: list[int],
    paper: float,
    *,
    allow_ink_rings: bool,
) -> list[tuple[int, int]]:
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = bbox
    sx0 = max(0, x0 - SEARCH_PAD_PX)
    sy0 = max(0, y0 - SEARCH_PAD_PX)
    sx1 = min(w, x1 + SEARCH_PAD_PX)
    sy1 = min(h, y1 + SEARCH_PAD_PX)
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


def _update_template_bbox(
    template: TemplateSpec,
    field_id: str,
    pixel_bbox: list[int],
) -> TemplateSpec:
    w, h = float(template.width), float(template.height)
    rel = [
        round(pixel_bbox[0] / w, 6),
        round(pixel_bbox[1] / h, 6),
        round(pixel_bbox[2] / w, 6),
        round(pixel_bbox[3] / h, 6),
    ]
    fields: list[FieldSpec] = []
    for spec in template.fields:
        if spec.field_id == field_id and spec.field_type == "checkbox":
            fields.append(spec.model_copy(update={"bbox": rel, "pad": 0.0}))
        else:
            fields.append(spec)
    return template.model_copy(update={"fields": fields})


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
    """Validate 1b checkbox windows. One rematch; else HiTL. No tick labels."""
    canvas = page.canvas
    result = layout.result
    template = layout.sectioned
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY) if canvas.ndim == 3 else canvas
    h, w = gray.shape[:2]
    paper = float(np.percentile(gray, 90))

    crops = dict(result.checkbox_crops)
    centers = {fid: _center(c.canonical_bbox) for fid, c in crops.items()}
    n_ok = n_retry = n_hitl = n_skip = 0

    for fid, crop in list(crops.items()):
        x0, y0, x1, y1 = crop.canonical_bbox
        if window_ok(crop.raw_image):
            crops[fid] = _tag(crop, ok=True, hitl=False, status="ok", attempts=0)
            n_ok += 1
            continue
        # Large overlay cells (label + box) are not 1b squares; skip unless they look like a label we can rematch.
        large = min(x1 - x0, y1 - y0) >= OVERLAY_SKIP_PX
        if large and not looks_like_text_line(crop.raw_image):
            crops[fid] = _tag(crop, ok=True, hitl=False, status="skip", attempts=0)
            n_skip += 1
            continue

        allow_ink = _interior_dark_frac(crop.raw_image) >= 0.08
        candidates = _search_candidates(gray, crop.canonical_bbox, paper, allow_ink_rings=allow_ink)
        own = _center(crop.canonical_bbox)
        ranked = sorted(candidates, key=lambda p: _dist((p[0] + 9.0, p[1] + 9.0), own))
        chosen: list[int] | None = None
        for hx, hy in ranked:
            cand_c = (hx + 9.0, hy + 9.0)
            if _closer_to_other(cand_c, fid, centers) or _too_close_to_other(cand_c, fid, centers):
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
            template = _update_template_bbox(template, fid, chosen)
            n_retry += 1
        else:
            crops[fid] = _tag(crop, ok=False, hitl=True, status="hitl", attempts=1)
            n_hitl += 1

    extra = dict(result.extra or {})
    extra["crop_validate"] = {
        "n_ok": n_ok,
        "n_retry": n_retry,
        "n_hitl": n_hitl,
        "n_skip": n_skip,
    }
    result.checkbox_crops = crops
    result.extra = extra
    if draw_debug:
        result.debug_overlay = draw_overlay(canvas, result, template)
    return Block1cLayout(result=result, sectioned=template)

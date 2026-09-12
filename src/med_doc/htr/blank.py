"""Blank-template patches for Block 3 difference imaging (no PHI)."""

from __future__ import annotations

import cv2
import numpy as np
from functools import lru_cache

from med_doc.paths import SYNTHETIC_DIR
from med_doc.schemas import TemplateSpec


def render_blank_form(template: TemplateSpec) -> np.ndarray:
    """Canonical printed squares / bars only — not a clinic photo."""
    w, h = template.width, template.height
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(
        img,
        (int(0.02 * w), int(0.02 * h)),
        (int(0.98 * w), int(0.07 * h)),
        (30, 30, 30),
        -1,
    )
    cv2.line(img, (int(0.03 * w), int(0.90 * h)), (int(0.97 * w), int(0.90 * h)), (40, 40, 40), 4)
    for spec in template.checkbox_fields():
        x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
        cv2.rectangle(img, (x0, y0), (x1, y1), (150, 150, 150), 2)
    return img


@lru_cache(maxsize=4)
def blank_canvas(template_id: str, width: int, height: int) -> np.ndarray:
    png = SYNTHETIC_DIR / "lab_request_v0_blank.png"
    if png.exists() and template_id != "lab_request_v1":
        bgr = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[1] != width or rgb.shape[0] != height:
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            return rgb
    try:
        from med_doc.template import load_template

        spec = load_template()
        if spec.width == width and spec.height == height:
            return render_blank_form(spec)
    except Exception:
        pass
    dummy = TemplateSpec(
        template_id=template_id or "blank",
        canvas_size=[width, height],
        fields=[],
    )
    return render_blank_form(dummy)


def blank_patch(
    template: TemplateSpec,
    field_id: str,
    crop_hw: tuple[int, int],
    bbox: list[int] | None = None,
) -> np.ndarray | None:
    canvas = blank_canvas(template.template_id, template.width, template.height)
    spec = template.field_map().get(field_id)
    if bbox is not None:
        x0, y0, x1, y1 = (int(v) for v in bbox)
    elif spec is not None:
        x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=True)
    else:
        return None
    h, w = canvas.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, max(x0 + 1, x1)), min(h, max(y0 + 1, y1))
    patch = canvas[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    rh, rw = crop_hw
    if patch.shape[0] != rh or patch.shape[1] != rw:
        patch = cv2.resize(patch, (rw, rh), interpolation=cv2.INTER_LINEAR)
    return patch


def residual_against_blank(crop: np.ndarray, blank: np.ndarray) -> np.ndarray:
    """Non-negative residual: printed structure subtracts away; handwriting remains."""
    g = np.asarray(crop)
    b = np.asarray(blank)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    if b.shape[:2] != g.shape[:2]:
        b = cv2.resize(b, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    # Small translational register (crop vs printed patch).
    if min(g.shape) >= 8 and min(b.shape) >= 8:
        pad = 4
        padded = cv2.copyMakeBorder(b, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        if padded.shape[0] > g.shape[0] and padded.shape[1] > g.shape[1]:
            ncc = cv2.matchTemplate(padded.astype(np.float32), g.astype(np.float32), cv2.TM_CCOEFF_NORMED)
            _min, _max, _minloc, maxloc = cv2.minMaxLoc(ncc)
            x, y = maxloc
            b = padded[y : y + g.shape[0], x : x + g.shape[1]]
    g = g.astype(np.float32)
    b = b.astype(np.float32)
    return np.clip(b - g, 0.0, 255.0)

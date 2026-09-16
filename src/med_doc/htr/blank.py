"""Blank-template patches for Block 3 difference imaging (no PHI)."""

from __future__ import annotations

import cv2
import numpy as np
from functools import lru_cache

from med_doc.paths import SYNTHETIC_DIR
from med_doc.schemas import TemplateSpec


def render_blank_form(template: TemplateSpec, *, labels: bool = False) -> np.ndarray:
    """Printed squares always. Labels, section bars, and write-in rules when ``labels``.

    ECC matches square edges on a photographed grid that has no Hershey names
    (``labels=False``). HTR residual must subtract the printed names and
    underlines that sit inside handwriting windows (``labels=True``).
    """
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
    if labels and getattr(template, "sections", None):
        for sec in template.main_sections():
            x0, y0, x1, y1 = [int(round(v * s)) for v, s in zip(sec.bbox, (w, h, w, h))]
            y1 = min(h, max(y0 + 8, y0 + int(0.018 * h)))
            cv2.rectangle(img, (max(0, x0), max(0, y0)), (min(w, x1), y1), (35, 35, 35), -1)
            caption = (sec.label or sec.id or "")[:28]
            if caption:
                cv2.putText(
                    img,
                    caption,
                    (max(0, x0) + 4, min(h - 2, y0 + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
    for spec in template.checkbox_fields():
        x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
        cv2.rectangle(img, (x0, y0), (x1, y1), (150, 150, 150), 2)
        if not labels:
            continue
        label = (spec.label or spec.field_id).split("(")[0].strip()[:22]
        if label:
            cv2.putText(
                img,
                label,
                (x1 + 4, min(h - 2, y1 - 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (25, 25, 25),
                1,
                cv2.LINE_AA,
            )
    if labels:
        for spec in template.handwriting_fields():
            x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
            if spec.field_id.startswith("tube_") or spec.field_type == "handwriting_line":
                cv2.line(img, (x0, y1 - 3), (x1, y1 - 3), (80, 80, 80), 2)
            else:
                cv2.rectangle(img, (x0, y0), (x1, y1), (170, 170, 170), 1)
            caption = (spec.label or spec.field_id).split("(")[0].strip()[:22]
            if caption:
                cv2.putText(
                    img,
                    caption,
                    (x0 + 2, min(h - 2, y0 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (25, 25, 25),
                    1,
                    cv2.LINE_AA,
                )
    return img


@lru_cache(maxsize=4)
def blank_canvas(template_id: str, width: int, height: int) -> np.ndarray:
    from med_doc.paths import DEFAULT_TEMPLATE, V1_TEMPLATE
    from med_doc.template import load_template

    tid = template_id or ""
    # Never subtract the v0 digital blank from a v1 page (1754 vs 1720, different rows).
    if "v1" not in tid:
        png = SYNTHETIC_DIR / "lab_request_v0_blank.png"
        if png.exists():
            bgr = cv2.imread(str(png), cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb.shape[1] != width or rgb.shape[0] != height:
                    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
                return rgb
    path = V1_TEMPLATE if "v1" in tid else DEFAULT_TEMPLATE
    try:
        spec = load_template(path)
        canvas = render_blank_form(spec, labels=True)
        if canvas.shape[1] != width or canvas.shape[0] != height:
            canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
        return canvas
    except Exception:
        pass
    dummy = TemplateSpec(
        template_id=tid or "blank",
        canvas_size=[width, height],
        fields=[],
    )
    return render_blank_form(dummy, labels=True)


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
        pad = 6
        padded = cv2.copyMakeBorder(b, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        if padded.shape[0] > g.shape[0] and padded.shape[1] > g.shape[1]:
            ncc = cv2.matchTemplate(padded.astype(np.float32), g.astype(np.float32), cv2.TM_CCOEFF_NORMED)
            _min, _max, _minloc, maxloc = cv2.minMaxLoc(ncc)
            x, y = maxloc
            b = padded[y : y + g.shape[0], x : x + g.shape[1]]
    g = g.astype(np.float32)
    b = b.astype(np.float32)
    return np.clip(b - g, 0.0, 255.0)

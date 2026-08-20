"""Printed header territories (main = black bar, sub = bold subhead)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from med_doc.schemas import FieldSpec, SectionSpec, TemplateSpec

# Black-bar blocks. Nested entries are bold subheads + their JSON groups
# until the next subhead (or the next black bar).
_V0_PLAN: list[dict] = [
    {
        "id": "checkup_profile",
        "label": "CHECK-UP / PROFILE",
        "column": 0,
        "groups": ["health_check", "specialty_profile"],
        "subs": [
            {"id": "health_check", "label": "Health Check", "groups": ["health_check"]},
            {"id": "specialty_profile", "label": "Specialty Profile", "groups": ["specialty_profile"]},
        ],
    },
    {"id": "haematology", "label": "HAEMATOLOGY", "column": 0, "groups": ["haematology"]},
    {"id": "bone_nutrition", "label": "BONE / NUTRITION", "column": 0, "groups": ["bone_nutrition"]},
    {"id": "cardiovascular", "label": "CARDIOVASCULAR", "column": 0, "groups": ["cardiovascular"]},
    {
        "id": "clinical_chemistry",
        "label": "CLINICAL CHEMISTRY",
        "column": 1,
        "groups": ["diabetes", "renal", "liver", "lipid", "chemistry"],
        "subs": [
            {"id": "diabetes", "label": "Diabetes", "groups": ["diabetes"]},
            {"id": "renal", "label": "Renal Function Tests", "groups": ["renal"]},
            {"id": "liver", "label": "Liver Function Tests", "groups": ["liver"]},
            {"id": "lipid_chemistry", "label": "Lipid Tests", "groups": ["lipid", "chemistry"]},
        ],
    },
    {"id": "immunology", "label": "IMMUNOLOGY", "column": 2, "groups": ["immunology"]},
    {"id": "endocrinology", "label": "ENDOCRINOLOGY", "column": 2, "groups": ["endocrinology"]},
    {"id": "tumour_markers", "label": "TUMOR MARKERS", "column": 3, "groups": ["tumour_markers"]},
    {"id": "urine", "label": "URINE", "column": 3, "groups": ["urine"]},
    {"id": "stool", "label": "STOOL", "column": 3, "groups": ["stool"]},
    {"id": "miscellaneous", "label": "MISCELLANEOUS", "column": 3, "groups": ["miscellaneous"]},
    {"id": "others", "label": "OTHERS", "column": 3, "groups": ["write_in"], "field_ids": ["others"]},
    {"id": "office_use", "label": "Office use only", "column": None, "groups": ["tube", "office"]},
]

_V1_PLAN: list[dict] = [
    {
        "id": "checkup_profile",
        "label": "CHECK-UP / PROFILE",
        "column": 0,
        "groups": ["health_check", "specialty_profile"],
        "subs": [
            {"id": "health_check", "label": "Health Check", "groups": ["health_check"]},
            {"id": "specialty_profile", "label": "Specialty Profile", "groups": ["specialty_profile"]},
        ],
    },
    {"id": "haematology", "label": "HAEMATOLOGY", "column": 0, "groups": ["haematology"]},
    {"id": "bone_nutrition", "label": "BONE / NUTRITION", "column": 0, "groups": ["bone_nutrition"]},
    {"id": "cardiovascular", "label": "CARDIOVASCULAR", "column": 0, "groups": ["cardiovascular"]},
    {
        "id": "clinical_chemistry",
        "label": "CLINICAL CHEMISTRY",
        "column": 1,
        "groups": ["diabetes", "renal", "liver", "lipid", "chemistry"],
        "subs": [
            {"id": "diabetes", "label": "Diabetes", "groups": ["diabetes"]},
            {"id": "renal", "label": "Renal Function Tests", "groups": ["renal"]},
            {"id": "liver", "label": "Liver Function Tests", "groups": ["liver"]},
            {"id": "lipid_chemistry", "label": "Lipid Tests", "groups": ["lipid", "chemistry"]},
        ],
    },
    {"id": "molecular", "label": "MOLECULAR", "column": 1, "groups": ["molecular"]},
    {"id": "immunology", "label": "IMMUNOLOGY", "column": 2, "groups": ["immunology"]},
    {"id": "endocrinology", "label": "ENDOCRINOLOGY", "column": 2, "groups": ["endocrinology"]},
    {"id": "tumour_markers", "label": "TUMOR MARKERS", "column": 3, "groups": ["tumour_markers"]},
    {"id": "urine", "label": "URINE", "column": 3, "groups": ["urine"]},
    {"id": "stool", "label": "STOOL", "column": 3, "groups": ["stool"]},
    {"id": "miscellaneous", "label": "MISCELLANEOUS", "column": 3, "groups": ["miscellaneous"]},
    {"id": "others", "label": "OTHERS", "column": 3, "groups": ["write_in"], "field_ids": ["others"]},
    {"id": "office_use", "label": "Office use only", "column": None, "groups": ["tube", "office"]},
]

_MAIN_HEADER = 0.026
_SUB_HEADER = 0.011
_GAP = 0.003


def _gutter_centers(template: TemplateSpec) -> list[float]:
    guts = [lm for lm in template.landmarks if lm.kind == "column_gutter"]
    guts = sorted(guts, key=lambda g: g.bbox[0])
    return [0.5 * (g.bbox[0] + g.bbox[2]) for g in guts]


def _column_x(gutters: list[float], column: int | None) -> tuple[float, float]:
    if column is None or not gutters:
        return 0.008, 0.992
    left = max(0.004, gutters[column] - 0.028)
    right = (gutters[column + 1] - 0.010) if column + 1 < len(gutters) else 0.992
    return left, min(0.996, right)


def _fit_header_width(
    bar: tuple[int, int, int, int],
    col_x0: int,
    col_x1: int,
    page_w: int,
) -> tuple[int, int]:
    """Keep the tick-box (left) side; pull the text-side (right) inside the header."""
    bx0, _, bx1, _ = bar
    bx0 = max(bx0, col_x0)
    bx1 = min(bx1, col_x1)
    if bx1 - bx0 < 0.08 * page_w:
        bx0, bx1 = bar[0], bar[2]
    width = max(bx1 - bx0, 1)
    left_inset = int(round(0.02 * width))
    right_inset = max(int(round(0.14 * width)), int(0.028 * page_w))
    bx0 += left_inset
    bx1 -= right_inset
    min_w = int(0.10 * page_w)
    if bx1 - bx0 < min_w:
        bx1 = min(page_w, bx0 + min_w)
    return bx0, bx1


def _clear_right_overlap(
    snapped: dict[str, SectionSpec],
    by_col: dict[int, list[SectionSpec]],
    gap: float = 0.010,
) -> None:
    """Shrink a column's right edge so it does not cover the next column's tick boxes."""
    cols = sorted(by_col)
    for col, nxt in zip(cols, cols[1:]):
        lefts = [snapped[s.id] for s in by_col[col] if s.id in snapped]
        rights = [snapped[s.id] for s in by_col[nxt] if s.id in snapped]
        for left in lefts:
            for right in rights:
                if left.bbox[3] <= right.bbox[1] or right.bbox[3] <= left.bbox[1]:
                    continue
                limit = right.bbox[0] - gap
                if left.bbox[2] > limit > left.bbox[0] + 0.08:
                    snapped[left.id] = left.model_copy(
                        update={"bbox": _clamp_box(left.bbox[0], left.bbox[1], limit, left.bbox[3])}
                    )
                    left = snapped[left.id]


def _fields_in_groups(template: TemplateSpec, groups: list[str]) -> list:
    wanted = set(groups)
    return [f for f in template.fields if (f.group or "") in wanted]


def _span(
    template: TemplateSpec,
    groups: list[str],
    field_ids: list[str] | None = None,
) -> tuple[float, float] | None:
    if field_ids:
        by_id = template.field_map()
        members = [by_id[i] for i in field_ids if i in by_id]
    else:
        members = _fields_in_groups(template, groups)
    if not members:
        return None
    return min(f.bbox[1] for f in members), max(f.bbox[3] for f in members)


def _clamp_box(x0: float, y0: float, x1: float, y1: float) -> list[float]:
    x0 = min(max(0.0, x0), 0.998)
    y0 = min(max(0.0, y0), 0.998)
    x1 = min(max(x0 + 0.004, x1), 1.0)
    y1 = min(max(y0 + 0.004, y1), 1.0)
    return [round(x0, 6), round(y0, 6), round(x1, 6), round(y1, 6)]


def _header_bottom_px(
    gray: np.ndarray,
    bbox: list[float],
    paper: float,
    height: int,
    width: int,
) -> int:
    """Pixel Y just below the black header inside a snapped main."""
    x0, y0, x1, y1 = [int(round(v * s)) for v, s in zip(bbox, (width, height, width, height))]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    band_h = min(70, max(12, y1 - y0))
    strip = gray[y0 : y0 + band_h, x0:x1]
    if strip.size == 0:
        return y0 + 12
    row_dark = (strip < paper * 0.55).mean(axis=1)
    bottom = 12
    for i, dark in enumerate(row_dark):
        if dark > 0.36:
            bottom = i + 1
        elif i > 8:
            break
    return y0 + max(bottom, 12)


def _row_has_checkbox(gray: np.ndarray, y: int, x0: int, x1: int, paper: float) -> bool:
    width = max(x1 - x0, 1)
    x_end = x0 + max(24, int(0.22 * width))
    for x in range(x0 + 2, x_end, 4):
        crop = gray[max(0, y) : y + 18, x : x + 18]
        if crop.size < 80:
            continue
        cy, cx = crop.shape[0] // 2, crop.shape[1] // 2
        center = float(crop[max(0, cy - 2) : cy + 3, max(0, cx - 2) : cx + 3].mean())
        dark = float((crop < paper * 0.55).mean())
        if center > paper * 0.72 and 0.06 <= dark <= 0.50:
            return True
    return False


def _snap_sub_top(
    gray: np.ndarray,
    expected_y: int,
    x0: int,
    x1: int,
    paper: float,
    y_min: int,
    y_max: int,
    height: int,
) -> int:
    """Nudge a mapped sub top onto a nearby bold label row (no checkbox)."""
    window = 48
    lo = max(y_min, expected_y - window)
    hi = min(y_max - 6, expected_y + window)
    text0 = x0 + int(0.20 * max(x1 - x0, 1))
    text1 = x1 - 4
    best_y, best = expected_y, -1.0
    for y in range(lo, hi):
        band = gray[y : y + 10, text0:text1]
        if band.size == 0:
            continue
        dark = float((band < paper * 0.55).mean())
        if dark < 0.14 or dark > 0.55:
            continue
        if _row_has_checkbox(gray, y, x0, x1, paper):
            continue
        score = dark - 0.008 * max(0, y - expected_y) - 0.001 * max(0, expected_y - y)
        if score > best:
            best, best_y = score, y
    return int(np.clip(best_y, 0, height - 4))


def _snap_subs_in_main(
    gray: np.ndarray,
    parent: SectionSpec,
    orig_parent: SectionSpec,
    orig_subs: list[SectionSpec],
    paper: float,
    height: int,
    width: int,
) -> list[SectionSpec]:
    """Place cyan subs: start at the bold subhead, drop until the next subhead."""
    if not orig_subs:
        return []
    x0, y0, x1, y1 = [int(round(v * s)) for v, s in zip(parent.bbox, (width, height, width, height))]
    header_y = _header_bottom_px(gray, parent.bbox, paper, height, width)
    src_c0 = min(s.bbox[1] for s in orig_subs)
    src_c1 = orig_parent.bbox[3]
    dst_c0 = header_y / height
    dst_c1 = parent.bbox[3]
    content_src = [orig_parent.bbox[0], src_c0, orig_parent.bbox[2], src_c1]
    content_dst = [parent.bbox[0], dst_c0, parent.bbox[2], dst_c1]
    mapped = []
    for spec in orig_subs:
        box = _map_box_into(spec.bbox, content_src, content_dst)
        mapped.append((spec, box))
    tops: list[int] = []
    for spec, box in mapped:
        expected = int(round(box[1] * height))
        tops.append(
            _snap_sub_top(gray, expected, x0, x1, paper, header_y, y1, height)
        )
    for i in range(1, len(tops)):
        if tops[i] <= tops[i - 1] + 8:
            tops[i] = tops[i - 1] + 8
    out: list[SectionSpec] = []
    inner_x0 = parent.bbox[0] + 0.002
    inner_x1 = parent.bbox[2] - 0.006
    for i, ((spec, _), top) in enumerate(zip(mapped, tops)):
        if i + 1 < len(tops):
            bot = tops[i + 1]
        else:
            bot = y1 - 2
        if bot <= top + 6:
            bot = top + 8
        out.append(
            spec.model_copy(
                update={"bbox": _clamp_box(inner_x0, top / height, inner_x1, bot / height)}
            )
        )
    return out


def _hollow_score(gray: np.ndarray, y: int, x0: int, x1: int, paper: float) -> float:
    best = 0.0
    x_lo = max(0, x0 - 16)
    x_hi = min(gray.shape[1] - 16, x0 + max(28, int(0.30 * max(x1 - x0, 1))))
    for x in range(x_lo, x_hi, 2):
        crop = gray[max(0, y) : y + 16, x : x + 16]
        if crop.shape[0] < 12 or crop.shape[1] < 12:
            continue
        cy, cx = crop.shape[0] // 2, crop.shape[1] // 2
        center = float(crop[cy - 2 : cy + 3, cx - 2 : cx + 3].mean())
        dark = float((crop < paper * 0.55).mean())
        if center > paper * 0.70 and 0.07 <= dark <= 0.48:
            score = dark * (center / max(paper, 1.0))
            # Prefer the top of the square (paper just above) so rows don't sit low.
            if y >= 3:
                above = float(gray[y - 3 : y, x : x + 16].mean())
                if above > paper * 0.80:
                    score *= 1.25
            best = max(best, score)
    return best


def _peak_indices(score: np.ndarray, *, min_dist: int, thresh: float) -> list[int]:
    out: list[int] = []
    i = 0
    n = len(score)
    while i < n:
        if score[i] < thresh:
            i += 1
            continue
        best = i
        j = i
        while j < n and (j - i) < min_dist and score[j] >= thresh * 0.55:
            if score[j] > score[best]:
                best = j
            j += 1
        out.append(best)
        i = best + min_dist
    return out


def detect_printed_rows(
    gray: np.ndarray,
    bbox: list[float],
    paper: float,
    height: int,
    width: int,
    *,
    skip_top: int = 16,
) -> list[int]:
    """Return pixel Y tops of printed checkbox rows inside a subsection (or leaf main)."""
    x0, y0, x1, y1 = [int(round(v * s)) for v, s in zip(bbox, (width, height, width, height))]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    ys, ye = y0 + skip_top, y1 - 4
    if ye - ys < 20:
        return []
    min_dist = max(24, int(0.015 * height))
    chk = np.zeros(ye - ys, dtype=float)
    for i, y in enumerate(range(ys, ye)):
        chk[i] = _hollow_score(gray, y, x0, x1, paper)
    return [ys + i for i in _peak_indices(chk, min_dist=min_dist, thresh=0.07)]


def count_printed_rows(
    gray: np.ndarray,
    bbox: list[float],
    paper: float,
    height: int,
    width: int,
    *,
    skip_top: int = 16,
) -> int:
    """Independent count: text-line peaks in the subsection body."""
    x0, y0, x1, y1 = [int(round(v * s)) for v, s in zip(bbox, (width, height, width, height))]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    ys, ye = y0 + skip_top, y1 - 4
    if ye - ys < 20:
        return 0
    min_dist = max(20, int(0.014 * height))
    text0 = x0 + int(0.18 * max(x1 - x0, 1))
    text1 = max(text0 + 8, x1 - 4)
    score = np.zeros(ye - ys, dtype=float)
    for i, y in enumerate(range(ys, ye)):
        td = float((gray[y : y + 8, text0:text1] < paper * 0.55).mean())
        if 0.10 <= td <= 0.55:
            score[i] = td
    return len(_peak_indices(score, min_dist=min_dist, thresh=0.12))


def _rows_in_leaf(
    gray: np.ndarray,
    leaf: SectionSpec,
    orig_leaf: SectionSpec,
    fields: list,
    paper: float,
    height: int,
    width: int,
) -> list[SectionSpec]:
    members = [f for f in fields if (f.group or "") in set(leaf.groups)]
    members.sort(key=lambda f: (f.bbox[1], f.bbox[0]))
    y0_px = int(round(leaf.bbox[1] * height))
    if leaf.kind == "sub":
        skip = 22
    else:
        skip = max(
            16,
            _header_bottom_px(gray, leaf.bbox, paper, height, width) - y0_px + 2,
        )
    peaks = detect_printed_rows(gray, leaf.bbox, paper, height, width, skip_top=skip)
    x0, _, x1, y1 = leaf.bbox
    y1_px = int(round(y1 * height))
    n = len(members)
    tops: list[int] = []
    if n and len(peaks) >= n:
        # Reading order, not snap_overlay Y: mapped field Y is biased down and
        # skips the first printed peak (urine / immunology / endocrinology / misc).
        tops = [int(p) for p in peaks[:n]]
    elif n and peaks:
        pitch = (
            int(np.median(np.diff(np.asarray(peaks, dtype=float))))
            if len(peaks) >= 2
            else max(24, int(0.018 * height))
        )
        tops = [int(p) for p in peaks]
        while len(tops) < n:
            nxt = tops[-1] + max(12, pitch)
            tops.append(int(min(nxt, y1_px - 8)))
    else:
        for field in members:
            mapped = _map_box_into(field.bbox, orig_leaf.bbox, leaf.bbox)
            tops.append(int(round(mapped[1] * height)))
    for i in range(1, len(tops)):
        if tops[i] <= tops[i - 1] + 8:
            tops[i] = tops[i - 1] + 12
    rows: list[SectionSpec] = []
    for i, top in enumerate(tops):
        bot = tops[i + 1] if i + 1 < len(tops) else y1_px - 2
        if bot <= top + 8:
            bot = top + 12
        rows.append(
            SectionSpec(
                id=f"{leaf.id}_row_{i + 1:02d}",
                kind="row",
                label=members[i].label if i < len(members) else "",
                bbox=_clamp_box(x0 + 0.004, top / height, x1 - 0.008, bot / height),
                column=leaf.column,
                parent_id=leaf.id,
                groups=list(leaf.groups),
                field_id=members[i].field_id if i < len(members) else None,
            )
        )
    return rows


def _ink_span(col_dark: np.ndarray, start: int, thresh: float) -> tuple[int, int] | None:
    hits = np.where(col_dark[start:] > thresh)[0]
    if hits.size < 3:
        return None
    return start + int(hits[0]), start + int(hits[-1]) + 1


def _hollow_xs(gray: np.ndarray, y0: int, y1: int, x0: int, x1: int, paper: float) -> list[int]:
    """Left edges of hollow-square ticks in a row strip, in page X."""
    found: list[int] = []
    y = max(0, y0 + max(0, (y1 - y0 - 16) // 2))
    x = x0
    while x <= x1 - 16:
        if _hollow_score(gray, y, x, x + 18, paper) >= 0.08:
            if not found or x - found[-1] >= 14:
                found.append(x)
            x += 14
        else:
            x += 2
    return found


def split_row_text_and_ticks(
    gray: np.ndarray,
    row: SectionSpec,
    paper: float,
    height: int,
    width: int,
    *,
    wall_x0: float | None = None,
) -> tuple[list[SectionSpec], list[SectionSpec]]:
    """Detect the test-name box in a row; leftover to the column wall is the tick box."""
    x0, y0, x1, y1 = [int(round(v * s)) for v, s in zip(row.bbox, (width, height, width, height))]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    wall = int(round((wall_x0 if wall_x0 is not None else row.bbox[0]) * width))
    wall = max(0, min(wall, x0))
    strip = gray[y0:y1, x0:x1]
    if strip.size == 0 or min(strip.shape) < 6:
        return [], []
    rh, rw = strip.shape
    col_dark = (strip < paper * 0.58).mean(axis=0)
    tick_zone = min(max(16, int(0.12 * rw)), max(8, rw // 3))
    hollow = _hollow_xs(gray, y0, y1, x0, x0 + min(rw, int(0.40 * rw)), paper)
    if hollow:
        text_start = min(rw - 4, max(tick_zone, hollow[0] - x0 + 16))
    else:
        text_start = tick_zone
    span = _ink_span(col_dark, text_start, 0.10)
    if span is None:
        span = _ink_span(col_dark, min(text_start, rw // 4), 0.08)
    texts: list[SectionSpec] = []
    ticks: list[SectionSpec] = []
    if span is not None:
        tx0, tx1 = span
        texts.append(
            SectionSpec(
                id=f"{row.id}_text",
                kind="text",
                label=row.label,
                bbox=_clamp_box((x0 + tx0) / width, y0 / height, (x0 + tx1) / width, y1 / height),
                column=row.column,
                parent_id=row.id,
                groups=list(row.groups),
                field_id=row.field_id,
            )
        )
        rem_x1 = x0 + max(8, tx0 - 1)
    else:
        rem_x1 = x0 + tick_zone
    # Tick spans from the column wall to the text box.
    if rem_x1 > wall + 4:
        ticks.append(
            SectionSpec(
                id=f"{row.id}_tick",
                kind="tick",
                label="",
                bbox=_clamp_box(wall / width, y0 / height, rem_x1 / width, y1 / height),
                column=row.column,
                parent_id=row.id,
                groups=list(row.groups),
                field_id=row.field_id,
            )
        )
    return texts, ticks


def _square_in_tick(
    gray: np.ndarray,
    tick: SectionSpec,
    paper: float,
    height: int,
    width: int,
) -> list[float] | None:
    """Return a ~one-square bbox inside the red tick strip, or None."""
    x0, y0, x1, y1 = [int(round(v * s)) for v, s in zip(tick.bbox, (width, height, width, height))]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 12 or y1 - y0 < 8:
        return None
    xs = _hollow_xs(gray, y0, y1, x0, x1, paper)
    if not xs:
        return None
    hx = int(xs[0])
    side = max(14, min(22, y1 - y0, x1 - hx))
    return _clamp_box(hx / width, y0 / height, (hx + side) / width, (y0 + side) / height)


def apply_tick_windows(
    canvas: np.ndarray,
    template: TemplateSpec,
) -> tuple[TemplateSpec, dict[str, Any]]:
    """Snap each checkbox field to the hollow square inside its red tick strip."""
    from med_doc.normalization.align import _checkbox_grid_score

    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY) if canvas.ndim == 3 else canvas
    h, w = gray.shape
    paper = float(np.percentile(gray, 90))
    by_field: dict[str, SectionSpec] = {}
    for spec in template.tick_sections():
        if spec.field_id:
            by_field[spec.field_id] = spec
    n_applied = 0
    fields: list[FieldSpec] = []
    for spec in template.fields:
        if spec.field_type != "checkbox":
            fields.append(spec)
            continue
        tick = by_field.get(spec.field_id)
        if tick is None:
            fields.append(spec)
            continue
        square = _square_in_tick(gray, tick, paper, h, w)
        if square is None:
            fields.append(spec)
            continue
        fields.append(spec.model_copy(update={"bbox": square}))
        n_applied += 1
    gated = template.model_copy(update={"fields": fields})
    grid = float(_checkbox_grid_score(gray, gated))
    meta: dict[str, Any] = {
        "method": "tick-windows",
        "n_ticks": len(by_field),
        "n_applied": n_applied,
        "grid": round(grid, 4),
    }
    return gated, meta


def _plan_for(template: TemplateSpec) -> list[dict]:
    tid = template.template_id
    if "v1" in tid:
        return _V1_PLAN
    return _V0_PLAN


def build_sections(template: TemplateSpec) -> list[SectionSpec]:
    """Derive main/sub territories from field groups and the printed-header plan."""
    gutters = _gutter_centers(template)
    plan = _plan_for(template)
    sections: list[SectionSpec] = []

    by_col: dict[int | None, list[dict]] = {}
    for main in plan:
        by_col.setdefault(main.get("column"), []).append(main)

    for column, mains in by_col.items():
        x0, x1 = _column_x(gutters, column)
        spans: list[tuple[dict, float, float]] = []
        for main in mains:
            span = _span(template, main["groups"], main.get("field_ids"))
            if span is None:
                continue
            spans.append((main, span[0], span[1]))
        spans.sort(key=lambda item: item[1])
        for idx, (main, y_content0, y_content1) in enumerate(spans):
            y0 = y_content0 - _MAIN_HEADER
            if idx + 1 < len(spans):
                y1 = spans[idx + 1][1] - _MAIN_HEADER - _GAP
            else:
                y1 = min(0.995, y_content1 + 0.012)
            if column is None:
                y0 = max(y0, 0.88)
            sections.append(
                SectionSpec(
                    id=main["id"],
                    kind="main",
                    label=main["label"],
                    bbox=_clamp_box(x0, y0, x1, y1),
                    column=column,
                    groups=list(main["groups"]),
                )
            )
            subs = main.get("subs") or []
            for sub_i, sub in enumerate(subs):
                sub_span = _span(template, sub["groups"])
                if sub_span is None:
                    continue
                sy0 = sub_span[0] - _SUB_HEADER
                if sub_i + 1 < len(subs):
                    nxt = _span(template, subs[sub_i + 1]["groups"])
                    sy1 = (nxt[0] - _SUB_HEADER - _GAP) if nxt else sub_span[1] + 0.006
                else:
                    sy1 = min(y1 - 0.002, sub_span[1] + 0.008)
                sy0 = max(sy0, y0 + 0.004)
                sy1 = min(sy1, y1 - 0.002)
                if sy1 <= sy0:
                    continue
                sections.append(
                    SectionSpec(
                        id=sub["id"],
                        kind="sub",
                        label=sub["label"],
                        bbox=_clamp_box(x0 + 0.006, sy0, x1 - 0.006, sy1),
                        column=column,
                        parent_id=main["id"],
                        groups=list(sub["groups"]),
                    )
                )
    return sections


def attach_sections(template: TemplateSpec) -> TemplateSpec:
    if template.sections:
        return template
    return template.model_copy(update={"sections": build_sections(template)})


def _bar_width(strip: np.ndarray, y0: int, y1: int, paper: float, x_abs0: int) -> tuple[int, int]:
    band = strip[y0:y1]
    if band.size == 0:
        return x_abs0, x_abs0 + strip.shape[1]
    dark = (band < paper * 0.50).mean(axis=0)
    hits = np.where(dark >= 0.40)[0]
    if hits.size < 4:
        return x_abs0, x_abs0 + strip.shape[1]
    return x_abs0 + int(hits[0]), x_abs0 + int(hits[-1]) + 1


def detect_header_bars(
    gray: np.ndarray,
    x0: int,
    x1: int,
    *,
    y_min: float,
    y_max: float,
) -> list[tuple[int, int, int, int]]:
    """Return (x0, y0, x1, y1) pixel boxes of black header bars in a column strip."""
    h, w = gray.shape
    xa, xb = max(0, x0), min(w, x1)
    if xb - xa < 16:
        return []
    strip = gray[:, xa:xb]
    paper = float(np.percentile(strip, 90))
    row_dark = (strip < paper * 0.55).mean(axis=1)
    row_mean = strip.mean(axis=1)
    raw: list[tuple[int, int, int, int]] = []
    in_bar = False
    y_start = 0
    height = strip.shape[0]
    for y in range(height):
        is_bar = bool(row_dark[y] > 0.36 and row_mean[y] < paper * 0.70)
        if is_bar and not in_bar:
            in_bar = True
            y_start = y
        elif not is_bar and in_bar:
            in_bar = False
            if (y - y_start) >= 3 and y_min < (y_start + y) / 2 < y_max:
                bx0, bx1 = _bar_width(strip, y_start, y, paper, xa)
                raw.append((bx0, y_start, bx1, y))
    if in_bar:
        y = height
        if (y - y_start) >= 3 and y_min < (y_start + y) / 2 < y_max:
            bx0, bx1 = _bar_width(strip, y_start, y, paper, xa)
            raw.append((bx0, y_start, bx1, y))
    kept: list[tuple[int, int, int, int]] = []
    for bx0, by0, bx1, by1 in _merge_close_bars(raw):
        if 12 <= (by1 - by0) <= 80 and (bx1 - bx0) >= 0.05 * w:
            kept.append((bx0, by0, bx1, by1))
    return kept


def _merge_close_bars(bars: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not bars:
        return []
    out = [bars[0]]
    for bar in bars[1:]:
        prev = out[-1]
        if bar[1] - prev[3] <= 28:
            out[-1] = (
                min(prev[0], bar[0]),
                prev[1],
                max(prev[2], bar[2]),
                max(prev[3], bar[3]),
            )
        else:
            out.append(bar)
    return out


def _map_box_into(
    box: list[float],
    src: list[float],
    dst: list[float],
) -> list[float]:
    sx0, sy0, sx1, sy1 = src
    dx0, dy0, dx1, dy1 = dst
    sw, sh = max(sx1 - sx0, 1e-6), max(sy1 - sy0, 1e-6)
    dw, dh = dx1 - dx0, dy1 - dy0
    fx0 = (box[0] - sx0) / sw
    fy0 = (box[1] - sy0) / sh
    fx1 = (box[2] - sx0) / sw
    fy1 = (box[3] - sy0) / sh
    return _clamp_box(dx0 + fx0 * dw, dy0 + fy0 * dh, dx0 + fx1 * dw, dy0 + fy1 * dh)


def _assign_bars_to_mains(
    mains: list[SectionSpec],
    bars: list[tuple[int, int, int, int]],
    height: int,
) -> list[tuple[SectionSpec, tuple[int, int, int, int]]]:
    """Pair the largest consecutive set of mains to bars whose Y spacing matches."""
    if not mains or not bars:
        return []
    expected = np.array(
        [0.5 * (s.bbox[1] + min(s.bbox[1] + 0.03, s.bbox[3])) * height for s in mains],
        dtype=float,
    )
    detected = np.array([(b[1] + b[3]) / 2.0 for b in bars], dtype=float)
    n_m, n_b = len(mains), len(bars)
    if n_b == n_m:
        return list(zip(mains, bars))
    for k in range(min(n_m, n_b), 0, -1):
        best: list[tuple[SectionSpec, tuple[int, int, int, int]]] | None = None
        best_err = 1e18
        for mi in range(n_m - k + 1):
            for bi in range(n_b - k + 1):
                exp = expected[mi : mi + k]
                det = detected[bi : bi + k]
                if k == 1:
                    err = float(abs(det[0] - exp[0]))
                    if err > 120:
                        continue
                else:
                    e0 = exp - exp[0]
                    d0 = det - det[0]
                    scale = float(np.dot(d0, e0) / (np.dot(e0, e0) + 1e-6))
                    if not (0.70 <= scale <= 1.40):
                        continue
                    err = float(np.max(np.abs(d0 - scale * e0)))
                    if err > 90:
                        continue
                if err < best_err:
                    best_err = err
                    best = [(mains[mi + i], bars[bi + i]) for i in range(k)]
        if best is not None:
            return best
    return []


def snap_sections(
    canvas: np.ndarray,
    template: TemplateSpec,
) -> tuple[TemplateSpec, dict[str, Any]]:
    """Snap each main section to a printed black header; height runs to the next header.

    Width follows the detected header bar, inset on the text side. Sub-sections snap
    onto bold subheads inside the locked main and drop until the next subhead.
    """
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY) if canvas.ndim == 3 else canvas
    h, w = gray.shape
    paper = float(np.percentile(gray, 90))
    gutters = _gutter_centers(template)
    by_id = {s.id: s for s in template.sections}
    mains = [s for s in template.main_sections() if s.column is not None]
    meta: dict[str, Any] = {"method": "header-to-next", "n_snapped": 0, "columns": []}
    if not mains or not gutters:
        return template, meta

    snapped_mains: dict[str, SectionSpec] = {}
    by_col: dict[int, list[SectionSpec]] = {}
    for spec in mains:
        by_col.setdefault(int(spec.column), []).append(spec)

    y_min, y_max = 0.0, 0.92 * h
    for col, col_mains in by_col.items():
        col_mains = sorted(col_mains, key=lambda s: s.bbox[1])
        left, right = _column_x(gutters, col)
        col_x0 = max(0, int(left * w))
        # Template columns overlap ~0.018; stop before the next column's tick boxes.
        fit_right = right
        if col + 1 < len(gutters):
            next_left = max(0.004, gutters[col + 1] - 0.028)
            fit_right = min(right, next_left - 0.012)
        col_x1 = min(w, int(fit_right * w))
        # Search a bit wider than the template column so a shifted bar is still seen.
        px0 = max(0, int((left - 0.02) * w))
        px1 = min(w, int((right + 0.02) * w))
        bars = detect_header_bars(gray, px0, px1, y_min=y_min, y_max=y_max)
        pairs = _assign_bars_to_mains(col_mains, bars, h)
        entry: dict[str, Any] = {
            "column": col,
            "n_bars": len(bars),
            "n_mains": len(col_mains),
            "n_paired": len(pairs),
            "fitted": bool(pairs),
        }
        if not pairs:
            meta["columns"].append(entry)
            continue
        for i, (spec, bar) in enumerate(pairs):
            bx0, by0, bx1, by1 = bar
            bx0, bx1 = _fit_header_width(bar, col_x0, col_x1, w)
            next_y = None
            if i + 1 < len(pairs):
                next_y = pairs[i + 1][1][1]
            else:
                later = [b for b in bars if b[1] > by1 + 2]
                if later:
                    next_y = later[0][1]
            y1 = int(next_y) if next_y is not None else int(0.888 * h)
            if y1 <= by0 + 8:
                y1 = by0 + 12
            snapped_mains[spec.id] = spec.model_copy(
                update={"bbox": _clamp_box(bx0 / w, by0 / h, bx1 / w, y1 / h)}
            )
            meta["n_snapped"] += 1
        meta["columns"].append(entry)

    _clear_right_overlap(snapped_mains, by_col)

    orig_subs_by_parent: dict[str, list[SectionSpec]] = {}
    for spec in template.sub_sections():
        orig_subs_by_parent.setdefault(spec.parent_id or "", []).append(spec)
    for subs in orig_subs_by_parent.values():
        subs.sort(key=lambda s: s.bbox[1])

    new_sections: list[SectionSpec] = []
    for spec in template.main_sections():
        if spec.column is None:
            new_sections.append(spec)
        else:
            new_sections.append(snapped_mains.get(spec.id, spec))

    n_subs = 0
    snapped_subs_all: list[SectionSpec] = []
    for parent_id, orig_subs in orig_subs_by_parent.items():
        parent = snapped_mains.get(parent_id)
        orig_parent = by_id.get(parent_id)
        if parent is None or orig_parent is None:
            new_sections.extend(orig_subs)
            continue
        snapped_subs = _snap_subs_in_main(
            gray, parent, orig_parent, orig_subs, paper, h, w
        )
        new_sections.extend(snapped_subs)
        snapped_subs_all.extend(snapped_subs)
        n_subs += len(snapped_subs)
    meta["n_subs_snapped"] = n_subs

    leaves: list[SectionSpec] = list(snapped_subs_all)
    mains_with_subs = {s.parent_id for s in snapped_subs_all}
    for spec in new_sections:
        if spec.kind != "main" or spec.column is None:
            continue
        if spec.id in mains_with_subs:
            continue
        if spec.id in {"others", "office_use"}:
            continue
        leaves.append(spec)

    n_rows = 0
    row_counts: list[dict[str, Any]] = []
    checkbox_fields = template.checkbox_fields()
    for leaf in leaves:
        orig_leaf = by_id.get(leaf.id, leaf)
        rows = _rows_in_leaf(gray, leaf, orig_leaf, checkbox_fields, paper, h, w)
        new_sections.extend(rows)
        n_rows += len(rows)
        skip = 22 if leaf.kind == "sub" else int(0.024 * h)
        printed = len(detect_printed_rows(gray, leaf.bbox, paper, h, w, skip_top=skip))
        expected = sum(1 for f in checkbox_fields if (f.group or "") in set(leaf.groups))
        row_counts.append(
            {
                "id": leaf.id,
                "kind": leaf.kind,
                "overlay": len(rows),
                "printed": printed,
                "template_fields": expected,
                "match": len(rows) == expected,
            }
        )
    meta["n_rows"] = n_rows
    meta["row_counts"] = row_counts

    n_text = 0
    n_tick = 0
    by_id_now = {s.id: s for s in new_sections}
    for spec in list(new_sections):
        if spec.kind != "row":
            continue
        parent = by_id_now.get(spec.parent_id)
        wall_src = parent
        while wall_src is not None and wall_src.kind == "sub" and wall_src.parent_id:
            nxt = by_id_now.get(wall_src.parent_id)
            if nxt is None:
                break
            wall_src = nxt
        wall_x0 = wall_src.bbox[0] if wall_src is not None else spec.bbox[0]
        texts, ticks = split_row_text_and_ticks(
            gray, spec, paper, h, w, wall_x0=wall_x0
        )
        new_sections.extend(texts)
        new_sections.extend(ticks)
        n_text += len(texts)
        n_tick += len(ticks)
    meta["n_text"] = n_text
    meta["n_tick"] = n_tick
    meta["confidence"] = float(meta["n_snapped"] / max(len(mains), 1))
    return template.model_copy(update={"sections": new_sections}), meta

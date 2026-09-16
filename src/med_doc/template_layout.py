"""Static layout audit for lab-request templates (header rows + label neighbors).

Independent of Block 1c. Catches authoring bugs that look like valid dark
crops because the window sits on a section bar or the wrong printed name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from med_doc.normalization.sections import _V0_PLAN, _V1_PLAN
from med_doc.schemas import FieldSpec, TemplateSpec

ROW_TOL = 0.45
SAME_ROW_Y = 0.004
# Other sub→sub / main→main gaps on v1 (specialty, lipid, urine, stool, …).
TYPICAL_SUB_GAP = 0.0296
TYPICAL_MAIN_GAP = 0.0285
LABEL_STRIP_FRAC = 0.11


@dataclass(frozen=True)
class HeaderField:
    field_id: str
    header_id: str
    header_label: str
    kind: str  # main | sub
    column: int
    groups: tuple[str, ...]


@dataclass(frozen=True)
class TransitionIssue:
    field_id: str
    header_label: str
    kind: str
    rows_off: float
    gap: float
    expected_gap: float
    note: str


def _plan_for(template: TemplateSpec) -> list[dict]:
    tid = template.template_id or ""
    return _V1_PLAN if "v1" in tid else _V0_PLAN


def _gutter_xs(template: TemplateSpec) -> list[float]:
    guts = [lm for lm in template.landmarks if lm.kind == "column_gutter"]
    guts = sorted(guts, key=lambda g: g.bbox[0])
    return [0.5 * (g.bbox[0] + g.bbox[2]) for g in guts]


def column_of(spec: FieldSpec, template: TemplateSpec) -> int:
    xs = _gutter_xs(template)
    cx = 0.5 * (spec.bbox[0] + spec.bbox[2])
    if not xs:
        return 0
    return int(min(range(len(xs)), key=lambda i: abs(cx - xs[i])))


def median_pitch(template: TemplateSpec, column: int | None = None) -> float:
    by: dict[str, list[FieldSpec]] = {}
    for spec in template.checkbox_fields():
        if column is not None and column_of(spec, template) != column:
            continue
        by.setdefault(spec.group or "", []).append(spec)
    diffs: list[float] = []
    for members in by.values():
        ys = sorted(0.5 * (f.bbox[1] + f.bbox[3]) for f in members)
        for a, b in zip(ys, ys[1:]):
            d = b - a
            if d > SAME_ROW_Y:
                diffs.append(d)
    if not diffs:
        return 0.02
    diffs.sort()
    return float(diffs[len(diffs) // 2])


def header_transition_fields(template: TemplateSpec) -> list[HeaderField]:
    """First checkbox after each printed main bar or bold subhead, reading order."""
    by_group: dict[str, list[FieldSpec]] = {}
    for spec in template.checkbox_fields():
        by_group.setdefault(spec.group or "", []).append(spec)
    for members in by_group.values():
        members.sort(key=lambda f: (f.bbox[1], f.bbox[0]))
    out: list[HeaderField] = []
    for block in _plan_for(template):
        col = block.get("column")
        if col is None:
            continue
        subs = block.get("subs") or []
        if subs:
            for sub in subs:
                groups = tuple(sub.get("groups") or [])
                first = _first_in_groups(by_group, groups)
                if first is None:
                    continue
                out.append(
                    HeaderField(
                        field_id=first.field_id,
                        header_id=str(sub["id"]),
                        header_label=str(sub.get("label") or sub["id"]),
                        kind="sub",
                        column=int(col),
                        groups=groups,
                    )
                )
        else:
            groups = tuple(block.get("groups") or [])
            first = _first_in_groups(by_group, groups)
            if first is None:
                continue
            out.append(
                HeaderField(
                    field_id=first.field_id,
                    header_id=str(block["id"]),
                    header_label=str(block.get("label") or block["id"]),
                    kind="main",
                    column=int(col),
                    groups=groups,
                )
            )
    return out


def _first_in_groups(by_group: dict[str, list[FieldSpec]], groups: tuple[str, ...]) -> FieldSpec | None:
    members: list[FieldSpec] = []
    for g in groups:
        members.extend(by_group.get(g) or [])
    if not members:
        return None
    members.sort(key=lambda f: (f.bbox[1], f.bbox[0]))
    return members[0]


def _last_y1_before(template: TemplateSpec, nxt: HeaderField, seen: list[HeaderField]) -> float | None:
    """Bottom of the previous block in the same column."""
    prev = [h for h in seen if h.column == nxt.column]
    if not prev:
        return None
    last = prev[-1]
    by_id = template.field_map()
    bottoms: list[float] = []
    for spec in template.checkbox_fields():
        if (spec.group or "") in last.groups:
            bottoms.append(spec.bbox[3])
        if spec.field_id == last.field_id:
            bottoms.append(spec.bbox[3])
    if not bottoms and last.field_id in by_id:
        bottoms.append(by_id[last.field_id].bbox[3])
    return max(bottoms) if bottoms else None


def audit_header_transitions(template: TemplateSpec) -> list[TransitionIssue]:
    """Flag first-after-header fields whose gap is ~one data row off."""
    fields = header_transition_fields(template)
    by_id = template.field_map()
    issues: list[TransitionIssue] = []
    seen: list[HeaderField] = []
    for item in fields:
        spec = by_id.get(item.field_id)
        if spec is None:
            seen.append(item)
            continue
        prev_y1 = _last_y1_before(template, item, seen)
        seen.append(item)
        if prev_y1 is None:
            continue
        pitch = median_pitch(template, item.column)
        expected = TYPICAL_MAIN_GAP if item.kind == "main" else TYPICAL_SUB_GAP
        gap = spec.bbox[1] - prev_y1
        rows = (gap - expected) / max(pitch, 1e-6)
        if abs(rows) < ROW_TOL:
            continue
        direction = "low (toward next row)" if rows > 0 else "high (on/into the header)"
        issues.append(
            TransitionIssue(
                field_id=item.field_id,
                header_label=item.header_label,
                kind=item.kind,
                rows_off=round(float(rows), 2),
                gap=round(float(gap), 5),
                expected_gap=expected,
                note=f"{item.field_id} after {item.header_label!r} is one-row {direction}",
            )
        )
    return issues


def label_tokens(label: str) -> list[str]:
    stripped = re.sub(r"\([^)]*\)", " ", label or "")
    parts = re.findall(r"[A-Za-z0-9]+", stripped)
    return [p.upper() for p in parts if len(p) >= 2]


def label_strip_px(template: TemplateSpec, spec: FieldSpec) -> list[int]:
    x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
    w, h = template.width, template.height
    pad_y = max(2, int(0.35 * (y1 - y0)))
    x_left = x1 + 3
    x_right = min(w, x1 + max(48, int(LABEL_STRIP_FRAC * w)))
    return [
        max(0, x_left),
        max(0, y0 - pad_y),
        max(x_left + 8, x_right),
        min(h, y1 + pad_y),
    ]


def _ncc_label(strip: np.ndarray, token: str) -> float:
    gray = strip if strip.ndim == 2 else cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
    if gray.size == 0 or min(gray.shape[:2]) < 8:
        return 0.0
    paper = 255
    canvas = np.full((48, max(64, 16 * len(token) + 24)), paper, dtype=np.uint8)
    cv2.putText(
        canvas,
        token,
        (4, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        20,
        1,
        cv2.LINE_AA,
    )
    templ = canvas
    best = 0.0
    gh, gw = gray.shape[:2]
    for scale in (0.55, 0.7, 0.9, 1.1, 1.35):
        th, tw = max(8, int(templ.shape[0] * scale)), max(12, int(templ.shape[1] * scale))
        if th >= gh or tw >= gw:
            continue
        small = cv2.resize(templ, (tw, th), interpolation=cv2.INTER_AREA)
        ncc = cv2.matchTemplate(gray.astype(np.float32), small.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        best = max(best, float(ncc.max()) if ncc.size else 0.0)
    return best


def label_neighbor_match(
    image: np.ndarray,
    template: TemplateSpec,
    spec: FieldSpec,
    *,
    min_score: float = 0.35,
) -> dict[str, Any]:
    """Does the strip immediately right of the checkbox contain the printed name?"""
    x0, y0, x1, y1 = label_strip_px(template, spec)
    crop = np.asarray(image)[y0:y1, x0:x1]
    tokens = label_tokens(spec.label or spec.field_id)
    scores = {tok: round(_ncc_label(crop, tok), 3) for tok in tokens[:4]}
    hit = [tok for tok, s in scores.items() if s >= min_score]
    ok = bool(hit) or (len(tokens) == 0)
    return {
        "field_id": spec.field_id,
        "label": spec.label,
        "tokens": tokens,
        "hit": hit,
        "scores": scores,
        "ok": ok,
        "strip": [x0, y0, x1, y1],
    }


def audit_label_neighbors(
    image: np.ndarray,
    template: TemplateSpec,
    *,
    field_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    wanted = set(field_ids) if field_ids else {h.field_id for h in header_transition_fields(template)}
    by_id = template.field_map()
    rows = []
    for fid in wanted:
        spec = by_id.get(fid)
        if spec is None or spec.field_type != "checkbox":
            continue
        rows.append(label_neighbor_match(image, template, spec))
    return rows


def looks_like_htr_garbage(text: str) -> bool:
    """Charset soup from printed form ink (batchsample7 others / office_other)."""
    raw = (text or "").strip()
    if not raw:
        return False
    compact = re.sub(r"[^A-Za-z0-9]+", "", raw)
    if compact.isdigit() and 1 <= len(compact) <= 4 and "\n" not in raw:
        return False
    if re.search(r"[./:\-]", raw) and not re.search(r"\d{1,2}[./\- ]\d{1,2}", raw):
        if len(compact) <= 4:
            return True
    tokens = [t for t in re.split(r"\s+", raw) if t]
    if tokens:
        skinny = sum(1 for t in tokens if len(re.sub(r"[^A-Za-z0-9]", "", t)) <= 1)
        if skinny / len(tokens) >= 0.55:
            return True
    if compact.upper() in {"AFP", "CEA", "CBC", "HBA1C", "LIPID", "URIC", "ALT", "AST", "TSH", "PSA", "CA125", "GLUCOSE"}:
        return False
    voiced = [
        t
        for t in tokens
        if re.search(r"[AEIOUaeiou]", t) and len(re.sub(r"[^A-Za-z]", "", t)) >= 3
    ]
    if not voiced and len(compact) >= 5:
        return True
    return len(compact) <= 2

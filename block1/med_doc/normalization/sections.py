"""Printed header territories (main = black bar, sub = bold subhead)."""

from __future__ import annotations

from med_doc.schemas import SectionSpec, TemplateSpec

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
    {"id": "others", "label": "OTHERS", "column": 3, "groups": ["write_in"]},
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
    {"id": "others", "label": "OTHERS", "column": 3, "groups": ["write_in"]},
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


def _fields_in_groups(template: TemplateSpec, groups: list[str]) -> list:
    wanted = set(groups)
    return [f for f in template.fields if (f.group or "") in wanted]


def _span(template: TemplateSpec, groups: list[str]) -> tuple[float, float] | None:
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
            span = _span(template, main["groups"])
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

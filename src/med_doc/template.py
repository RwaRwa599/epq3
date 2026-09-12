from __future__ import annotations

from pathlib import Path

from med_doc.paths import DEFAULT_TEMPLATE
from med_doc.schemas import TemplateSpec
from med_doc.normalization.sections import attach_sections


def load_template(path: str | Path | None = None) -> TemplateSpec:
    p = Path(path) if path else DEFAULT_TEMPLATE
    spec = TemplateSpec.model_validate_json(p.read_text(encoding="utf-8"))
    return attach_sections(spec)

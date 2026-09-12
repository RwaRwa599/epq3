"""Crop-level gold for checkbox ticks. Field ids only — no images in git."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SheetGold(BaseModel):
    """One sheet: which checkbox field ids are actually ticked."""

    doc_id: str
    ticked_field_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class GoldSet(BaseModel):
    version: str = "1.0"
    kind: str = "checkbox_ticks"
    sheets: list[SheetGold] = Field(default_factory=list)

    def by_doc(self) -> dict[str, SheetGold]:
        return {s.doc_id: s for s in self.sheets}


def load_gold(path: str | Path) -> GoldSet:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "sheets" not in raw and "doc_id" in raw:
        return GoldSet(sheets=[SheetGold.model_validate(raw)])
    return GoldSet.model_validate(raw)


def dump_gold(gold: GoldSet | SheetGold, path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload: Any = gold
    if isinstance(gold, SheetGold):
        payload = GoldSet(sheets=[gold])
    dest.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    return dest

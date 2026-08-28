from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

FieldType = Literal["checkbox", "handwriting_box", "handwriting_line"]
LandmarkKind = Literal["header_bar", "footer_rule", "column_gutter"]
SectionKind = Literal["main", "sub", "row", "text", "tick"]


class FieldSpec(BaseModel):
    """One ROI on the canonical page, in relative [0, 1] coordinates."""

    field_id: str
    field_type: FieldType
    bbox: list[float] = Field(min_length=4, max_length=4)
    label: str = ""
    group: str | None = None
    pad: float = 0.002

    @field_validator("bbox")
    @classmethod
    def _unit_square(cls, value: list[float]) -> list[float]:
        if not all(0.0 <= v <= 1.0 for v in value):
            raise ValueError("bbox coordinates must lie in [0, 1]")
        if value[2] <= value[0] or value[3] <= value[1]:
            raise ValueError("bbox must have x1 > x0 and y1 > y0")
        return value


class LandmarkSpec(BaseModel):
    id: str
    kind: LandmarkKind
    bbox: list[float] = Field(min_length=4, max_length=4)

    @field_validator("bbox")
    @classmethod
    def _unit_square(cls, value: list[float]) -> list[float]:
        if not all(0.0 <= v <= 1.0 for v in value):
            raise ValueError("landmark bbox coordinates must lie in [0, 1]")
        return value


class SectionSpec(BaseModel):
    """A printed header territory on the canonical page (relative [0, 1] bbox)."""

    id: str
    kind: SectionKind
    label: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    column: int | None = None
    parent_id: str | None = None
    groups: list[str] = Field(default_factory=list)
    field_id: str | None = None

    @field_validator("bbox")
    @classmethod
    def _unit_square(cls, value: list[float]) -> list[float]:
        if not all(0.0 <= v <= 1.0 for v in value):
            raise ValueError("bbox coordinates must lie in [0, 1]")
        if value[2] <= value[0] or value[3] <= value[1]:
            raise ValueError("bbox must have x1 > x0 and y1 > y0")
        return value


class TemplateSpec(BaseModel):
    template_id: str
    canvas_size: list[int] = Field(min_length=2, max_length=2)
    fields: list[FieldSpec]
    landmarks: list[LandmarkSpec] = Field(default_factory=list)
    handwriting_order: list[str] = Field(default_factory=list)
    sections: list[SectionSpec] = Field(default_factory=list)

    @property
    def width(self) -> int:
        return int(self.canvas_size[0])

    @property
    def height(self) -> int:
        return int(self.canvas_size[1])

    def field_map(self) -> dict[str, FieldSpec]:
        return {f.field_id: f for f in self.fields}

    def checkbox_fields(self) -> list[FieldSpec]:
        return [f for f in self.fields if f.field_type == "checkbox"]

    def handwriting_fields(self) -> list[FieldSpec]:
        by_id = self.field_map()
        if self.handwriting_order:
            return [by_id[i] for i in self.handwriting_order if i in by_id]
        return [f for f in self.fields if f.field_type != "checkbox"]

    def main_sections(self) -> list[SectionSpec]:
        return [s for s in self.sections if s.kind == "main"]

    def sub_sections(self, parent_id: str | None = None) -> list[SectionSpec]:
        subs = [s for s in self.sections if s.kind == "sub"]
        if parent_id is None:
            return subs
        return [s for s in subs if s.parent_id == parent_id]

    def row_sections(self, parent_id: str | None = None) -> list[SectionSpec]:
        rows = [s for s in self.sections if s.kind == "row"]
        if parent_id is None:
            return rows
        return [s for s in rows if s.parent_id == parent_id]

    def text_sections(self, parent_id: str | None = None) -> list[SectionSpec]:
        items = [s for s in self.sections if s.kind == "text"]
        if parent_id is None:
            return items
        return [s for s in items if s.parent_id == parent_id]

    def tick_sections(self, parent_id: str | None = None) -> list[SectionSpec]:
        items = [s for s in self.sections if s.kind == "tick"]
        if parent_id is None:
            return items
        return [s for s in items if s.parent_id == parent_id]

    def pixel_bbox(self, spec: FieldSpec, *, apply_pad: bool = True) -> list[int]:
        w, h = self.width, self.height
        x0, y0, x1, y1 = spec.bbox
        pad = spec.pad if apply_pad else 0.0
        return [
            int(round((x0 - pad) * w)),
            int(round((y0 - pad) * h)),
            int(round((x1 + pad) * w)),
            int(round((y1 + pad) * h)),
        ]


class FieldCrop(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    field_id: str
    field_type: FieldType
    canonical_bbox: list[int]
    normalized_image: np.ndarray
    raw_image: np.ndarray
    quality_score: float = Field(ge=0.0, le=1.0)
    blur_score: float = Field(ge=0.0, le=1.0, default=0.0)
    glare_index: float = Field(ge=0.0, le=1.0, default=0.0)


class SectionCrop(BaseModel):
    """Tick-column extra-ink crop for one main or sub (raw pixels, no blank subtraction)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    section_id: str
    canonical_bbox: list[int]
    raw_image: np.ndarray
    normalized_image: np.ndarray
    field_ids: list[str] = Field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=1.0, default=0.0)


class NormalizedDocumentResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    document_id: str
    canonical_canvas: np.ndarray
    alignment_confidence: float = Field(ge=0.0, le=1.0)
    checkbox_crops: dict[str, FieldCrop]
    handwriting_crops: dict[str, FieldCrop]
    section_crops: dict[str, SectionCrop] = Field(default_factory=dict)
    debug_overlay: np.ndarray | None = None
    warp_method: str = "none"
    orientation_degrees: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)

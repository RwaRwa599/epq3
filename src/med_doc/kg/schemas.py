"""Data contracts and schemas for Block 2 Clinical Knowledge Graph."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class CatalogueItem(BaseModel):
    """Metadata for a single clinical test or profile entry."""

    field_id: str
    label: str
    section: str = "general"
    kind: Literal["test", "profile", "write_in"] = "test"
    aliases: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    tubes: list[str] = Field(default_factory=list)
    loinc: str | None = None
    description: str | None = None


class RankedCandidate(BaseModel):
    """A scored hypothesis for handwriting recognition or candidate ranking."""

    value: str
    canonical_id: str | None = None
    score: float = Field(ge=0.0, le=1.0)
    tier: Literal[1, 2, 3] = 1
    reason: str = "direct_match"


class ValidationResult(BaseModel):
    """Report generated from cross-checking observations against medical rules."""

    is_valid: bool
    ticked_tests: list[str] = Field(default_factory=list)
    implied_tests: list[str] = Field(default_factory=list)
    all_ordered_tests: list[str] = Field(default_factory=list)
    expected_tubes: dict[str, int] = Field(default_factory=dict)
    observed_tubes: dict[str, int | None] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    discrepancies: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

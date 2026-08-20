"""Data contracts for Block 3 handwriting recognition and mark classification."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MarkPrediction(BaseModel):
    """Classification of a single checkbox crop."""

    field_id: str
    is_marked: bool
    confidence: float = Field(ge=0.0, le=1.0)
    ink_density: float = Field(ge=0.0, le=1.0, default=0.0)
    needs_hitl: bool = False
    source: str = "density"


class HandwritingPrediction(BaseModel):
    """Recognition result for a single handwriting crop."""

    field_id: str
    raw_text: str = ""
    canonical_value: str | None = None
    canonical_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    source: str = "empty"
    needs_hitl: bool = True
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)


class DocumentPrediction(BaseModel):
    """Per-document Block 3 output for LIS / Block 4."""

    doc_id: str
    is_valid: bool = True
    checkbox_marks: dict[str, MarkPrediction] = Field(default_factory=dict)
    handwriting_fields: dict[str, HandwritingPrediction] = Field(default_factory=dict)
    ticked_test_ids: list[str] = Field(default_factory=list)
    implied_tests: list[str] = Field(default_factory=list)
    expected_tubes: dict[str, int] = Field(default_factory=dict)
    observed_tubes: dict[str, int | None] = Field(default_factory=dict)
    discrepancies: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    hitl_fields: list[str] = Field(default_factory=list)


class DocumentHypotheses(BaseModel):
    """Per-document verbal + nonverbal hypotheses for Block 4 / Colab tables."""

    version: str = "1.0"
    block: str = "block3"
    doc_id: str
    nonverbal: dict[str, MarkPrediction] = Field(default_factory=dict)
    verbal: dict[str, HandwritingPrediction] = Field(default_factory=dict)
    ticked_test_ids: list[str] = Field(default_factory=list)
    implied_tests: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    hitl_fields: list[str] = Field(default_factory=list)


class BatchPredictionManifest(BaseModel):
    """Batch summary for a Block 3 ZIP export."""

    version: str = "1.0"
    block: str = "block3"
    stage: str = "verbal_nonverbal"
    total_documents: int = 0
    valid_documents: int = 0
    hitl_documents: int = 0
    documents: list[dict[str, Any]] = Field(default_factory=list)

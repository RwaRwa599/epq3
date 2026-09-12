"""Block 5 contracts: HiTL queue, review patches, LIS order."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ReviewAction = Literal[
    "confirm_tick",
    "reject_tick",
    "set_text",
    "set_tube",
    "accept_write_in",
]


class ReviewPatch(BaseModel):
    """One human (or scripted) override. Applied to Block 3 drafts, then re-scored."""

    field_id: str
    action: ReviewAction
    value: str | None = None
    actor: str = "reviewer"
    ts: str | None = None


class HitlItem(BaseModel):
    """One field the LIS must not commit without a look."""

    field_id: str
    kind: Literal["tick", "tube", "write_in", "date", "other"] = "other"
    reason: str = ""
    crop_path: str | None = None
    raw_text: str = ""
    nbest: list[str] = Field(default_factory=list)
    is_marked: bool | None = None
    canonical_id: str | None = None
    llm_suggestions: list[dict[str, Any]] = Field(default_factory=list)


class DocumentReview(BaseModel):
    version: str = "1.0"
    block: str = "block5"
    doc_id: str
    patches: list[ReviewPatch] = Field(default_factory=list)
    queue: list[HitlItem] = Field(default_factory=list)
    auto_passed: bool = False


class LabOrder(BaseModel):
    """Generic lab-order payload. Clinic adapters map this, they do not re-fuse."""

    version: str = "1.0"
    block: str = "block5"
    doc_id: str
    ordered_tests: list[str] = Field(default_factory=list)
    ticked_test_ids: list[str] = Field(default_factory=list)
    implied_tests: list[str] = Field(default_factory=list)
    observed_tubes: dict[str, int | None] = Field(default_factory=dict)
    expected_tubes: dict[str, int] = Field(default_factory=dict)
    received_at: str | None = None
    others_raw: str | None = None
    others_canonical_id: str | None = None
    needs_review: bool = False
    is_valid: bool = True
    discrepancies: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0, default=1.0)

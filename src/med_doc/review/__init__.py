"""Block 5: HiTL review, optional LLM suggestions, LIS commit."""

from med_doc.review.apply import apply_patches, commit_hypotheses
from med_doc.review.batch import process_from_block4
from med_doc.review.lis import order_from_prediction
from med_doc.review.llm import LlmRanker, ScriptedLlmRanker, attach_llm_suggestions
from med_doc.review.queue import build_hitl_queue
from med_doc.review.schemas import DocumentReview, HitlItem, LabOrder, ReviewPatch

__all__ = [
    "DocumentReview",
    "HitlItem",
    "LabOrder",
    "LlmRanker",
    "ReviewPatch",
    "ScriptedLlmRanker",
    "apply_patches",
    "attach_llm_suggestions",
    "build_hitl_queue",
    "commit_hypotheses",
    "order_from_prediction",
    "process_from_block4",
]

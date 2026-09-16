"""Block 5: HiTL review, optional LLM suggestions, LIS commit."""

from med_doc.review.apply import apply_patches, commit_hypotheses
from med_doc.review.batch import process_from_block4
from med_doc.review.lis import order_from_prediction
from med_doc.review.llm import LlmRanker, OllamaRanker, ScriptedLlmRanker, attach_llm_suggestions
from med_doc.review.queue import build_hitl_queue
from med_doc.review.sanity import REGISTRATION_FAILURE, crop_quality, diagnostic_confidence
from med_doc.review.schemas import DocumentReview, HitlItem, LabOrder, OrderBundle, OutputMode, ReviewPatch

__all__ = [
    "DocumentReview",
    "HitlItem",
    "LabOrder",
    "OrderBundle",
    "OutputMode",
    "LlmRanker",
    "OllamaRanker",
    "ReviewPatch",
    "ScriptedLlmRanker",
    "apply_patches",
    "attach_llm_suggestions",
    "build_hitl_queue",
    "commit_hypotheses",
    "order_from_prediction",
    "process_from_block4",
    "REGISTRATION_FAILURE",
    "crop_quality",
    "diagnostic_confidence",
]

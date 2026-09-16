"""Block 3: nonverbal marks (Paddle) and verbal handwriting (TrOCR)."""

from __future__ import annotations

from med_doc.htr.batch import process_batch_from_block2, process_document, process_from_block1
from med_doc.htr.fusion import fuse_handwriting
from med_doc.htr.marks import TICK_POLICY, classify_mark
from med_doc.htr.nonverbal import classify_mark_nonverbal, classify_marks, paddle_available
from med_doc.htr.recognizer import recognize_handwriting
from med_doc.htr.schemas import (
    BatchPredictionManifest,
    DocumentHypotheses,
    DocumentPrediction,
    HandwritingPrediction,
    MarkPrediction,
    VisionDraft,
)
from med_doc.htr.verbal import (
    attach_kg_priors,
    recognize_fields,
    recognize_verbal,
    trocr_available,
)
from med_doc.htr.vision import ollama_available, run_vision_draft, disagreement_crop_ids

__all__ = [
    "MarkPrediction",
    "HandwritingPrediction",
    "DocumentPrediction",
    "DocumentHypotheses",
    "BatchPredictionManifest",
    "classify_mark",
    "TICK_POLICY",
    "classify_mark_nonverbal",
    "classify_marks",
    "paddle_available",
    "recognize_handwriting",
    "recognize_verbal",
    "recognize_fields",
    "trocr_available",
    "ollama_available",
    "run_vision_draft",
    "disagreement_crop_ids",
    "VisionDraft",
    "attach_kg_priors",
    "fuse_handwriting",
    "process_document",
    "process_from_block1",
    "process_batch_from_block2",
]

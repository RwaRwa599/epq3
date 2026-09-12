"""Medical document intelligence — Block 3 nonverbal marks + verbal HTR."""

from med_doc.htr import (
    BatchPredictionManifest,
    DocumentHypotheses,
    DocumentPrediction,
    HandwritingPrediction,
    MarkPrediction,
    classify_mark,
    classify_mark_nonverbal,
    classify_marks,
    fuse_handwriting,
    process_batch_from_block2,
    process_document,
    process_from_block1,
    recognize_fields,
    recognize_handwriting,
    recognize_verbal,
)
from med_doc.kg import KnowledgeGraph

__all__ = [
    "KnowledgeGraph",
    "MarkPrediction",
    "HandwritingPrediction",
    "DocumentPrediction",
    "DocumentHypotheses",
    "BatchPredictionManifest",
    "classify_mark",
    "classify_mark_nonverbal",
    "classify_marks",
    "recognize_handwriting",
    "recognize_verbal",
    "recognize_fields",
    "fuse_handwriting",
    "process_document",
    "process_from_block1",
    "process_batch_from_block2",
]

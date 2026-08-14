"""Medical document intelligence — Block 3 Handwriting Recognition & Prior Fusion."""

from med_doc.htr import (
    BatchPredictionManifest,
    DocumentPrediction,
    HandwritingPrediction,
    MarkPrediction,
    classify_mark,
    fuse_handwriting,
    process_batch_from_block2,
    process_document,
    recognize_handwriting,
)
from med_doc.kg import KnowledgeGraph

__all__ = [
    "KnowledgeGraph",
    "MarkPrediction",
    "HandwritingPrediction",
    "DocumentPrediction",
    "BatchPredictionManifest",
    "classify_mark",
    "recognize_handwriting",
    "fuse_handwriting",
    "process_document",
    "process_batch_from_block2",
]

"""Block 3: Handwriting recognition, mark classification, and prior fusion."""

from __future__ import annotations

from med_doc.htr.batch import process_batch_from_block2, process_document
from med_doc.htr.fusion import fuse_handwriting
from med_doc.htr.marks import classify_mark
from med_doc.htr.recognizer import recognize_handwriting
from med_doc.htr.schemas import (
    BatchPredictionManifest,
    DocumentPrediction,
    HandwritingPrediction,
    MarkPrediction,
)

__all__ = [
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

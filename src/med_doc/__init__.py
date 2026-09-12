"""Medical document intelligence — Blocks 1–4 (normalize, KG, HTR, rescoring)."""

from med_doc.htr import (
    BatchPredictionManifest,
    DocumentHypotheses,
    DocumentPrediction,
    HandwritingPrediction,
    MarkPrediction,
    process_batch_from_block2,
    process_from_block1,
)
from med_doc.kg import (
    CatalogueItem,
    KnowledgeGraph,
    RankedCandidate,
    ValidationResult,
    process_batch_from_block1,
)
from med_doc.normalization import (
    normalize_batch,
    normalize_document,
    save_normalized_document,
)
from med_doc.rescoring import process_from_block3, rescore_hypotheses
from med_doc.schemas import FieldCrop, NormalizedDocumentResult, TemplateSpec

__all__ = [
    "FieldCrop",
    "NormalizedDocumentResult",
    "TemplateSpec",
    "normalize_document",
    "normalize_batch",
    "save_normalized_document",
    "KnowledgeGraph",
    "CatalogueItem",
    "RankedCandidate",
    "ValidationResult",
    "process_batch_from_block1",
    "MarkPrediction",
    "HandwritingPrediction",
    "DocumentPrediction",
    "BatchPredictionManifest",
    "process_batch_from_block2",
    "process_from_block1",
    "process_from_block3",
    "rescore_hypotheses",
    "DocumentHypotheses",
]

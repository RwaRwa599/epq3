"""Medical document intelligence — Block 1 Normalization & Block 2 Knowledge Graph."""

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
]

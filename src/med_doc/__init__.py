"""Medical document intelligence — Block 1 Normalization & Block 2 Knowledge Graph."""

from med_doc.kg import CatalogueItem, KnowledgeGraph, RankedCandidate, ValidationResult
from med_doc.normalization import normalize_document
from med_doc.schemas import FieldCrop, NormalizedDocumentResult, TemplateSpec

__all__ = [
    "FieldCrop",
    "NormalizedDocumentResult",
    "TemplateSpec",
    "normalize_document",
    "KnowledgeGraph",
    "CatalogueItem",
    "RankedCandidate",
    "ValidationResult",
]

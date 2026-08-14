"""Medical document intelligence — Block 1 Document Normalization & ROI Extraction."""

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
]

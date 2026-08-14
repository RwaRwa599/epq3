"""Medical document intelligence — Block 1 is document normalization and ROI extraction."""

from med_doc.normalization import normalize_document
from med_doc.schemas import FieldCrop, NormalizedDocumentResult, TemplateSpec

__all__ = [
    "FieldCrop",
    "NormalizedDocumentResult",
    "TemplateSpec",
    "normalize_document",
]

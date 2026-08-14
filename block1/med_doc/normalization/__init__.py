"""Normalization subpackage for document quad detection, warp, fine alignment, and batch processing."""

from med_doc.normalization.batch import normalize_batch, save_normalized_document
from med_doc.normalization.pipeline import normalize_document

__all__ = [
    "normalize_document",
    "normalize_batch",
    "save_normalized_document",
]

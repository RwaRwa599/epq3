"""Block 2: Clinical Knowledge Graph and Medical Prior Engine."""

from __future__ import annotations

from med_doc.kg.graph import KnowledgeGraph
from med_doc.kg.schemas import CatalogueItem, RankedCandidate, ValidationResult

__all__ = [
    "KnowledgeGraph",
    "CatalogueItem",
    "RankedCandidate",
    "ValidationResult",
]

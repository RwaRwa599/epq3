"""Experimental parallel pipeline (if1block1–4). Does not replace live Blocks 1–5."""

from med_doc.if1.if1block1 import normalize_batch, normalize_document
from med_doc.if1.if1block2 import estimate_groups_from_counts, load_groups, score
from med_doc.if1.if1block3 import process_from_if1block1
from med_doc.if1.if1block4 import process_from_if1block3
from med_doc.if1.pipeline import run_if1

__all__ = [
    "normalize_document",
    "normalize_batch",
    "load_groups",
    "score",
    "estimate_groups_from_counts",
    "process_from_if1block1",
    "process_from_if1block3",
    "run_if1",
]

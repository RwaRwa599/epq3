"""Text normalization and fuzzy similarity utilities for clinical entities."""

from __future__ import annotations

import difflib
import re
import unicodedata


def normalize_text(text: str | None) -> str:
    """Normalize text by lowercasing, removing diacritics, and collapsing spaces."""
    if not text:
        return ""
    # Normalize unicode (decompose accented chars, e.g. β-hCG -> b-hcg or keep greek)
    text = str(text).strip().lower()
    # Replace common medical symbols
    text = text.replace("β", "beta").replace("α", "alpha").replace("&", " and ")
    # Replace punctuation with space except alphanumeric and hyphens
    text = re.sub(r"[^\w\s-]", " ", text)
    # Collapse multiple whitespace / underscores
    text = re.sub(r"[\s_]+", " ", text).strip()
    return text


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Fast similarity ratio between two strings [0.0 - 1.0]."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def token_similarity(a: str, b: str) -> float:
    """Token-aware similarity combining exact token overlap and sequence matching."""
    norm_a = normalize_text(a)
    norm_b = normalize_text(b)
    if norm_a == norm_b:
        return 1.0
    if not norm_a or not norm_b:
        return 0.0

    seq_sim = levenshtein_ratio(norm_a, norm_b)

    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())
    if not tokens_a or not tokens_b:
        return seq_sim

    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    jaccard = inter / max(1, union)

    # If all tokens of one are in the other (e.g. 'cbc' in 'cbc with diff')
    if tokens_a.issubset(tokens_b) or tokens_b.issubset(tokens_a):
        subset_bonus = 0.85
    else:
        subset_bonus = 0.0

    return float(max(seq_sim, jaccard, subset_bonus))

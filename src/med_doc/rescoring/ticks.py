"""Trusted vs HiTL ticks. Uncertain marks never expand profiles or tubes."""

from __future__ import annotations

from med_doc.htr.schemas import MarkPrediction


def trusted_tick_ids(nonverbal: dict[str, MarkPrediction]) -> list[str]:
    """Marks that are ticked and not HiTL — the only set used for expected tubes."""
    return [fid for fid, mark in nonverbal.items() if mark.is_marked and not mark.needs_hitl]


def uncertain_tick_ids(nonverbal: dict[str, MarkPrediction]) -> list[str]:
    """Ticked but HiTL (crop gate, low confidence). Never used for tube priors."""
    return [fid for fid, mark in nonverbal.items() if mark.is_marked and mark.needs_hitl]


def split_nonverbal_ticks(
    nonverbal: dict[str, MarkPrediction],
) -> tuple[list[str], list[str]]:
    return trusted_tick_ids(nonverbal), uncertain_tick_ids(nonverbal)

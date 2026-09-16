"""Block 5 sanity: registration-failure gate (does not try to fix crops)."""

from __future__ import annotations

from med_doc.htr.quality import crop_quality, diagnostic_confidence

REGISTRATION_FAILURE = "registration_failure_suspected"
# order-10 inflated to 40–50+ ordered tests after profile expansion
MAX_ORDERED_TESTS = 40
# 40% of checkbox fields on the form (only when n_checkbox looks like a full sheet)
TICKED_OF_CATALOGUE = 0.40
# Clinic dumps at 98% retry still looked "fine" because this gate ignored crop_retry_rate.
CROP_RETRY_RATE_FAIL = 0.40
MIN_CHECKBOXES_FOR_RETRY = 40


def registration_failure_reasons(
    *,
    ticked_test_ids: list[str],
    ordered_tests: list[str],
    observed_tubes: dict[str, int | None],
    n_checkbox: int,
    all_tube_crops_empty: bool,
    n_implausible_tubes: int = 0,
    crop_retry_rate: float = 0.0,
) -> list[str]:
    """Independent of CV fixes: stop a fully-populated wrong order looking authoritative."""
    n_ticked = len(ticked_test_ids)
    n_ordered = len(ordered_tests)
    details: list[str] = []
    if n_ordered >= MAX_ORDERED_TESTS:
        details.append(
            f"{REGISTRATION_FAILURE}: {n_ordered} ordered tests (cap {MAX_ORDERED_TESTS})"
        )
    if n_checkbox >= 80 and n_checkbox > 0 and (n_ticked / n_checkbox) >= TICKED_OF_CATALOGUE:
        details.append(
            f"{REGISTRATION_FAILURE}: ticked {n_ticked} / {n_checkbox} checkboxes "
            f">= {TICKED_OF_CATALOGUE:.0%}"
        )
    retry = float(crop_retry_rate or 0.0)
    if n_checkbox >= MIN_CHECKBOXES_FOR_RETRY and retry >= CROP_RETRY_RATE_FAIL:
        details.append(
            f"{REGISTRATION_FAILURE}: crop_retry_rate {retry:.0%} "
            f"(cap {CROP_RETRY_RATE_FAIL:.0%})"
        )
    tube_obs = dict(observed_tubes or {})
    n_implausible = int(n_implausible_tubes or 0)
    tubes_empty = (not tube_obs) or all(v is None for v in tube_obs.values())
    if n_ticked > 0 and n_implausible > 0:
        details.append(
            f"{REGISTRATION_FAILURE}: {n_implausible} implausible tube count(s) rejected"
        )
    elif n_ticked > 0 and (tubes_empty or all_tube_crops_empty):
        details.append(f"{REGISTRATION_FAILURE}: all tube crops empty")
    return details


__all__ = [
    "REGISTRATION_FAILURE",
    "crop_quality",
    "diagnostic_confidence",
    "registration_failure_reasons",
]

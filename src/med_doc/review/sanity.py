"""Block 5 sanity: registration-failure gate (does not try to fix crops)."""

from __future__ import annotations

from med_doc.htr.quality import crop_quality, diagnostic_confidence

REGISTRATION_FAILURE = "registration_failure_suspected"
# 40–50% of ordered_tests (midpoint). Ignore tiny legitimate orders (CBC-only is 1/1).
TICKED_OF_ORDERED = 0.45
MIN_TICKS_FOR_RATIO = 8
# order-10 inflated to 40–50+ ordered tests after profile expansion
MAX_ORDERED_TESTS = 40
# 40% of checkbox fields on the form (only when n_checkbox looks like a full sheet)
TICKED_OF_CATALOGUE = 0.40


def registration_failure_reasons(
    *,
    ticked_test_ids: list[str],
    ordered_tests: list[str],
    observed_tubes: dict[str, int | None],
    n_checkbox: int,
    all_tube_crops_empty: bool,
    n_implausible_tubes: int = 0,
) -> list[str]:
    """Independent of CV fixes: stop a fully-populated wrong order looking authoritative."""
    n_ticked = len(ticked_test_ids)
    n_ordered = len(ordered_tests)
    details: list[str] = []
    if (
        n_ticked >= MIN_TICKS_FOR_RATIO
        and n_ordered > 0
        and (n_ticked / n_ordered) >= TICKED_OF_ORDERED
    ):
        details.append(
            f"{REGISTRATION_FAILURE}: ticked {n_ticked} / ordered {n_ordered} "
            f">= {TICKED_OF_ORDERED:.0%}"
        )
    if n_ordered >= MAX_ORDERED_TESTS:
        details.append(
            f"{REGISTRATION_FAILURE}: {n_ordered} ordered tests (cap {MAX_ORDERED_TESTS})"
        )
    if n_checkbox >= 80 and n_checkbox > 0 and (n_ticked / n_checkbox) >= TICKED_OF_CATALOGUE:
        details.append(
            f"{REGISTRATION_FAILURE}: ticked {n_ticked} / {n_checkbox} checkboxes "
            f">= {TICKED_OF_CATALOGUE:.0%}"
        )
    tube_obs = dict(observed_tubes or {})
    n_implausible = int(n_implausible_tubes or 0)
    if n_ticked > 0 and n_implausible > 0:
        details.append(
            f"{REGISTRATION_FAILURE}: {n_implausible} implausible tube count(s) rejected"
        )
    elif n_ticked > 0 and tube_obs and all(v is None for v in tube_obs.values()):
        details.append(f"{REGISTRATION_FAILURE}: all tube crops empty")
    elif n_ticked > 0 and all_tube_crops_empty:
        details.append(f"{REGISTRATION_FAILURE}: all tube crops empty")
    return details


__all__ = [
    "REGISTRATION_FAILURE",
    "crop_quality",
    "diagnostic_confidence",
    "registration_failure_reasons",
]

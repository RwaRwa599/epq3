"""if1block2 coverage scorer.

One formula for missing-likely, odd-member, and high/low order tiers.

Research placeholder: fitting (w, π) from counts / causal graphs is
``estimate_groups_from_counts`` — not implemented.
"""

from __future__ import annotations

from typing import Any

DEFAULT_PRIOR = 1.0
DEFAULT_TAU_MISS = 0.35
DEFAULT_TAU_ODD = 0.50


def group_coverage(group: dict[str, Any], ticked: set[str]) -> float:
    """s_g in [0, 1]. Ticking the group id itself counts as full coverage."""
    gid = str(group.get("id") or "")
    if gid and gid in ticked:
        return 1.0
    members = dict(group.get("members") or {})
    if not members:
        return 0.0
    weight = float(sum(float(w) for w in members.values()) or 0.0)
    if weight <= 0:
        return 0.0
    hit = sum(float(w) for tid, w in members.items() if tid in ticked)
    return float(hit / weight)


def score(
    ticked: list[str],
    groups: list[dict[str, Any]],
    *,
    catalogue: set[str] | None = None,
) -> dict[str, Any]:
    """Return per-group a_g and per-id m / e.

    s_g = coverage; a_g = π_g * s_g
    m(u) = max_g a_g w_{u,g} (1 - x_u)
    e(u) = x_u * (1 - max_g s_g w_{u,g})

    Extra score uses coverage s_g (not π_g) so a placeholder prior of 1.0
    does not demote in-group ticks. π_g scales missing inference only via a_g.
    """
    x = set(ticked)
    activations: dict[str, dict[str, float]] = {}
    max_support: dict[str, float] = {}
    max_missing: dict[str, float] = {}

    for group in groups:
        gid = str(group.get("id") or "")
        members = {str(k): float(v) for k, v in dict(group.get("members") or {}).items()}
        prior = float(group.get("prior") if group.get("prior") is not None else DEFAULT_PRIOR)
        s_g = group_coverage(group, x)
        a_g = prior * s_g
        activations[gid] = {"s": round(s_g, 4), "a": round(a_g, 4), "prior": prior}
        if gid:
            max_support[gid] = max(max_support.get(gid, 0.0), s_g)
            max_missing[gid] = max(max_missing.get(gid, 0.0), 0.0)
        for tid, w in members.items():
            if catalogue is not None and tid not in catalogue:
                continue
            support = s_g * w
            miss = a_g * w
            max_support[tid] = max(max_support.get(tid, 0.0), support)
            max_missing[tid] = max(max_missing.get(tid, 0.0), miss)

    ids = set(max_support) | set(max_missing) | x
    if catalogue is not None:
        ids = {i for i in ids if i in catalogue or i in x}

    m_scores: dict[str, float] = {}
    e_scores: dict[str, float] = {}
    for tid in ids:
        xu = 1.0 if tid in x else 0.0
        m_scores[tid] = round((1.0 - xu) * max_missing.get(tid, 0.0), 4)
        e_scores[tid] = round(xu * (1.0 - max_support.get(tid, 0.0)), 4)

    return {
        "activations": activations,
        "m": m_scores,
        "e": e_scores,
    }


def split_tiers(
    ticked: list[str],
    scored: dict[str, Any],
    *,
    tau_miss: float = DEFAULT_TAU_MISS,
    tau_odd: float = DEFAULT_TAU_ODD,
    still_ticked: list[str] | None = None,
) -> dict[str, list[str]]:
    """high = still ticked and e < tau_odd; low = missing ∪ extras."""
    e = dict(scored.get("e") or {})
    m = dict(scored.get("m") or {})
    marked = set(still_ticked if still_ticked is not None else ticked)
    initial = set(ticked)

    high = sorted(u for u in marked if float(e.get(u) or 0.0) < tau_odd)
    low_missing = sorted(
        u for u, val in m.items() if float(val) >= tau_miss and u not in high
    )
    low_extra = sorted(
        u
        for u in (marked | initial)
        if float(e.get(u) or 0.0) >= tau_odd and u not in high
    )
    low = sorted(set(low_missing) | set(low_extra))
    return {
        "ordered_tests_high": high,
        "ordered_tests_low": low,
        "low_missing": low_missing,
        "low_extra": low_extra,
    }


def flags_from_scores(
    scored: dict[str, Any],
    ticked: list[str],
    *,
    tau_miss: float = DEFAULT_TAU_MISS,
    tau_odd: float = DEFAULT_TAU_ODD,
    cap: int = 16,
) -> list[dict[str, Any]]:
    from med_doc.rescoring.combinations import CombinationFlag

    initial = set(ticked)
    flags: list[CombinationFlag] = []
    for tid, val in (scored.get("m") or {}).items():
        if tid in initial:
            continue
        if float(val) >= tau_miss:
            flags.append(
                CombinationFlag(tid, "missing_likely", float(val), reason="if1_coverage_missing")
            )
    for tid, val in (scored.get("e") or {}).items():
        if tid not in initial:
            continue
        if float(val) >= tau_odd:
            flags.append(
                CombinationFlag(tid, "odd_member", float(val), reason="if1_coverage_extra")
            )
    flags.sort(key=lambda f: (-f.confidence, f.kind, f.field_id))
    return flags[:cap]

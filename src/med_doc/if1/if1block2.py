"""if1block2: frozen named groups + coverage scorer.

Group topology is seeded from printed ``profile_bundles``. Fitting weights
from co-occurrence counts is a research placeholder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from med_doc.if1 import scorer as coverage
from med_doc.kg.graph import KnowledgeGraph
from med_doc.paths import DEFAULT_KG, IF1_GROUPS, IF1_LOCAL_OVERRIDES


def groups_from_profile_bundles(kg: KnowledgeGraph | None = None) -> list[dict[str, Any]]:
    kg = kg or KnowledgeGraph.load(DEFAULT_KG)
    groups: list[dict[str, Any]] = []
    for gid, members in (kg.profile_bundles or {}).items():
        if not members:
            continue
        item = kg.get_item(gid)
        groups.append(
            {
                "id": gid,
                "members": {str(m): 1.0 for m in members},
                "prior": coverage.DEFAULT_PRIOR,
                "diagnosis": (item.label if item is not None else gid),
            }
        )
    return groups


def _apply_overrides(
    groups: list[dict[str, Any]],
    overrides: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not overrides:
        return groups
    multipliers = dict(overrides.get("prior_multipliers") or {})
    disabled = set(overrides.get("disabled_groups") or [])
    out: list[dict[str, Any]] = []
    for group in groups:
        gid = str(group.get("id") or "")
        if gid in disabled:
            continue
        row = dict(group)
        if gid in multipliers:
            row["prior"] = float(row.get("prior") or coverage.DEFAULT_PRIOR) * float(
                multipliers[gid]
            )
        out.append(row)
    return out


def load_groups(
    path: str | Path | None = None,
    *,
    overrides_path: str | Path | None = None,
    apply_local: bool = False,
    kg: KnowledgeGraph | None = None,
) -> list[dict[str, Any]]:
    target = Path(path) if path is not None else IF1_GROUPS
    if target.is_file():
        payload = json.loads(target.read_text(encoding="utf-8"))
        groups = list(payload.get("groups") or payload if isinstance(payload, dict) else payload)
    else:
        groups = groups_from_profile_bundles(kg)
    if apply_local:
        opath = Path(overrides_path) if overrides_path is not None else IF1_LOCAL_OVERRIDES
        if opath.is_file():
            groups = _apply_overrides(groups, json.loads(opath.read_text(encoding="utf-8")))
    return groups


def load_kg(path: str | Path | None = None) -> KnowledgeGraph:
    return KnowledgeGraph.load(path or DEFAULT_KG)


def score(initial_ticked: list[str], *, groups: list[dict[str, Any]] | None = None, kg: KnowledgeGraph | None = None) -> dict[str, Any]:
    kg = kg or load_kg()
    groups = groups if groups is not None else load_groups(kg=kg)
    catalogue = set(kg.catalogue)
    return coverage.score(initial_ticked, groups, catalogue=catalogue)


def estimate_groups_from_counts(*_args, **_kwargs):
    """Fit (w, π) from tick-set counts.

    Placeholder for later research (Dirichlet/Beta shrinkage onto these presets,
    not unconstrained causal discovery or LLM-proposed edges).
    """
    raise NotImplementedError(
        "if1 group generation from counts is a research placeholder "
        "(Bayesian shrinkage / co-occurrence). Use frozen if1_groups_v0.json."
    )

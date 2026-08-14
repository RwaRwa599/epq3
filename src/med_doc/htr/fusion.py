"""Fuse raw HTR drafts with Block 2 Knowledge Graph priors."""

from __future__ import annotations

from typing import Any

from med_doc.htr.recognizer import extract_digits, parse_datetime
from med_doc.htr.schemas import HandwritingPrediction
from med_doc.kg.graph import KnowledgeGraph
from med_doc.kg.schemas import RankedCandidate
from med_doc.kg.textutil import token_similarity

DEFAULT_TAU = 0.75


def _candidates_from_priors(priors: list[dict[str, Any]] | None) -> list[RankedCandidate]:
    out: list[RankedCandidate] = []
    for row in priors or []:
        try:
            out.append(RankedCandidate(**row))
        except Exception:
            continue
    return out


def fuse_handwriting(
    field_id: str,
    draft_text: str,
    draft_conf: float,
    draft_source: str,
    *,
    prior_rankings: list[dict[str, Any]] | None = None,
    kg: KnowledgeGraph | None = None,
    ticked_ids: list[str] | None = None,
    expected_tubes: dict[str, int] | None = None,
    tau: float = DEFAULT_TAU,
) -> HandwritingPrediction:
    """Combine a raw HTR draft with Block 2 priors into a canonical prediction."""
    ticked_ids = ticked_ids or []
    expected_tubes = expected_tubes or {}
    priors = _candidates_from_priors(prior_rankings)
    hypotheses: list[dict[str, Any]] = [
        {"value": draft_text, "score": round(draft_conf, 3), "source": draft_source}
    ]

    # Tube counts: prefer extracted digits, else expected count from KG
    if field_id.startswith("tube_"):
        return _fuse_tube(
            field_id,
            draft_text,
            draft_conf,
            draft_source,
            kg=kg,
            ticked_ids=ticked_ids,
            expected_tubes=expected_tubes,
            priors=priors,
            hypotheses=hypotheses,
            tau=tau,
        )

    # Date / received-at fields
    if field_id in {"received_at", "date", "sample_received"}:
        parsed, pconf = parse_datetime(draft_text)
        if parsed:
            hypotheses.append({"value": parsed, "score": pconf, "source": "date_parse"})
            return HandwritingPrediction(
                field_id=field_id,
                raw_text=draft_text,
                canonical_value=parsed,
                canonical_id=field_id,
                confidence=round(pconf, 3),
                source="date_parse",
                needs_hitl=pconf < tau,
                hypotheses=hypotheses,
            )
        empty = not (draft_text or "").strip()
        return HandwritingPrediction(
            field_id=field_id,
            raw_text=draft_text,
            canonical_value=draft_text or None,
            canonical_id=field_id if draft_text else None,
            confidence=round(draft_conf if draft_text else 0.85, 3),
            source=draft_source,
            needs_hitl=not empty and draft_conf < tau,
            hypotheses=hypotheses,
        )

    # Free-text clinical_info / office_other: keep raw, flag if ink but unreadable
    if field_id in {"clinical_info", "office_other"}:
        empty = not (draft_text or "").strip()
        return HandwritingPrediction(
            field_id=field_id,
            raw_text=draft_text,
            canonical_value=draft_text or None,
            canonical_id=None,
            confidence=round(draft_conf if not empty else 0.85, 3),
            source=draft_source,
            needs_hitl=not empty and draft_conf < tau,
            hypotheses=hypotheses,
        )

    # OTHERS / write-in: match against catalogue + prior rankings
    return _fuse_write_in(
        field_id,
        draft_text,
        draft_conf,
        draft_source,
        kg=kg,
        ticked_ids=ticked_ids,
        priors=priors,
        hypotheses=hypotheses,
        tau=tau,
    )


def _fuse_tube(
    field_id: str,
    draft_text: str,
    draft_conf: float,
    draft_source: str,
    *,
    kg: KnowledgeGraph | None,
    ticked_ids: list[str],
    expected_tubes: dict[str, int],
    priors: list[RankedCandidate],
    hypotheses: list[dict[str, Any]],
    tau: float,
) -> HandwritingPrediction:
    digits = extract_digits(draft_text)
    obs: int | None = int(digits) if digits else None

    tube_name = ""
    expected = 0
    if kg is not None:
        tube_name = kg.tube_field_to_tube.get(field_id, "")
        if not expected_tubes:
            expected_tubes = kg.calculate_expected_tubes(ticked_ids)
        expected = int(expected_tubes.get(tube_name, 0))
    elif priors:
        try:
            expected = int(extract_digits(priors[0].value) or 0)
        except ValueError:
            expected = 0

    if obs is not None:
        value = str(obs)
        if obs == expected:
            conf, source, hitl = 0.95, f"{draft_source}+prior", False
        elif abs(obs - expected) <= 1:
            conf, source, hitl = 0.70, f"{draft_source}+near_prior", True
        else:
            conf, source, hitl = 0.45, draft_source, True
    elif expected > 0 and draft_source in {"empty", "ink-present"}:
        # No readable digit; if the crop was empty, report 0, else use prior
        if draft_source == "empty":
            value, conf, source, hitl = "0" if expected == 0 else str(expected), 0.6, "prior_expected", expected > 0
            if expected == 0:
                hitl = False
                conf = 0.9
                source = "empty"
                value = ""
        else:
            value, conf, source, hitl = str(expected), 0.55, "prior_expected", True
    else:
        value, conf, source, hitl = digits or "", round(draft_conf, 3), draft_source, bool(digits) and draft_conf < tau

    hypotheses.append({"value": value, "score": round(conf, 3), "source": source})
    return HandwritingPrediction(
        field_id=field_id,
        raw_text=draft_text,
        canonical_value=value or None,
        canonical_id=field_id if value else None,
        confidence=round(float(conf), 3),
        source=source,
        needs_hitl=bool(hitl),
        hypotheses=hypotheses,
    )


def _fuse_write_in(
    field_id: str,
    draft_text: str,
    draft_conf: float,
    draft_source: str,
    *,
    kg: KnowledgeGraph | None,
    ticked_ids: list[str],
    priors: list[RankedCandidate],
    hypotheses: list[dict[str, Any]],
    tau: float,
) -> HandwritingPrediction:
    empty = not (draft_text or "").strip()
    if empty and draft_source == "empty":
        return HandwritingPrediction(
            field_id=field_id,
            raw_text="",
            canonical_value=None,
            canonical_id=None,
            confidence=0.9,
            source="empty",
            needs_hitl=False,
            hypotheses=hypotheses,
        )

    ranked: list[RankedCandidate] = list(priors)
    if kg is not None and (draft_text or "").strip():
        ranked = kg.assume(field_id, draft_text, {"ticked_ids": ticked_ids}, top_k=5)
    elif kg is not None and not ranked:
        ranked = kg.fuzzy_match_catalogue(draft_text or "", top_k=5)

    best: RankedCandidate | None = ranked[0] if ranked else None
    if best is not None:
        hypotheses.append(
            {
                "value": best.value,
                "canonical_id": best.canonical_id,
                "score": best.score,
                "source": best.reason,
                "tier": best.tier,
            }
        )

    if best is not None and (draft_text or "").strip():
        sim = token_similarity(draft_text, best.value)
        score = max(best.score, sim)
        # High-tier catalogue hit with decent draft confidence
        if best.tier == 1 and score >= 0.72:
            conf = max(draft_conf, score)
            needs_hitl = conf < tau
            return HandwritingPrediction(
                field_id=field_id,
                raw_text=draft_text,
                canonical_value=best.value,
                canonical_id=best.canonical_id,
                confidence=round(min(1.0, conf), 3),
                source=f"{draft_source}+kg",
                needs_hitl=needs_hitl,
                hypotheses=hypotheses,
            )
        return HandwritingPrediction(
            field_id=field_id,
            raw_text=draft_text,
            canonical_value=best.value,
            canonical_id=best.canonical_id,
            confidence=round(min(1.0, score * 0.9), 3),
            source=f"{draft_source}+kg",
            needs_hitl=True,
            hypotheses=hypotheses,
        )

    if (draft_text or "").strip() and draft_conf >= tau:
        return HandwritingPrediction(
            field_id=field_id,
            raw_text=draft_text,
            canonical_value=draft_text,
            canonical_id=None,
            confidence=round(draft_conf, 3),
            source=draft_source,
            needs_hitl=False,
            hypotheses=hypotheses,
        )

    return HandwritingPrediction(
        field_id=field_id,
        raw_text=draft_text,
        canonical_value=draft_text or None,
        canonical_id=None,
        confidence=round(draft_conf, 3),
        source=draft_source,
        needs_hitl=not empty,
        hypotheses=hypotheses,
    )

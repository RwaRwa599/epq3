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
    extra_hypotheses: list[dict[str, Any]] | None = None,
    tau: float = DEFAULT_TAU,
) -> HandwritingPrediction:
    """Combine a raw HTR draft with Block 2 priors into a canonical prediction."""
    ticked_ids = ticked_ids or []
    expected_tubes = expected_tubes or {}
    priors = _candidates_from_priors(prior_rankings)
    hypotheses: list[dict[str, Any]] = [
        {"value": draft_text, "score": round(draft_conf, 3), "source": draft_source}
    ]
    for row in extra_hypotheses or []:
        if isinstance(row, dict) and row not in hypotheses:
            hypotheses.append(row)

    # Tube counts: digits from Block 3 drafts only — never fill empty crops from KG.
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
            extra_hypotheses=extra_hypotheses,
            tau=tau,
        )

    # Date / received-at fields
    if field_id in {"received_at", "date", "sample_received"}:
        parsed, pconf = parse_datetime(draft_text)
        if parsed and pconf >= 0.8:
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
            canonical_id=None,
            confidence=round(draft_conf if draft_text else 0.85, 3),
            source=draft_source,
            needs_hitl=not empty,
            hypotheses=hypotheses,
        )

    # Free-text clinical_info / office_other: keep raw, flag if ink but unreadable
    if field_id in {"clinical_info", "office_other"}:
        empty = not (draft_text or "").strip()
        engine_gap = draft_source in {"unavailable", "ink-present", "trocr-nodigit"}
        return HandwritingPrediction(
            field_id=field_id,
            raw_text=draft_text,
            canonical_value=draft_text or None,
            canonical_id=None,
            confidence=round(draft_conf if not empty else (0.0 if engine_gap else 0.85), 3),
            source=draft_source,
            needs_hitl=engine_gap or (not empty and draft_conf < tau),
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
        extra_hypotheses=extra_hypotheses,
        tau=tau,
    )


def _digits_from_drafts(draft_text: str, extra_hypotheses: list[dict[str, Any]] | None) -> str:
    digits = extract_digits(draft_text)
    if digits:
        return digits
    for row in extra_hypotheses or []:
        if not isinstance(row, dict):
            continue
        alt = extract_digits(str(row.get("value") or row.get("text") or ""))
        if alt:
            return alt
    return ""


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
    extra_hypotheses: list[dict[str, Any]] | None = None,
    tau: float,
) -> HandwritingPrediction:
    digits = _digits_from_drafts(draft_text, extra_hypotheses)
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
    else:
        # Empty / unreadable crop: observed is missing. Never copy expected into LIS.
        value = ""
        source = draft_source or "empty"
        if source == "empty" and expected <= 0:
            conf, hitl = 0.9, False
        else:
            conf = round(draft_conf, 3) if draft_conf else 0.3
            hitl = expected > 0 or source in {
                "ink-present",
                "unavailable",
                "trocr-nodigit",
                "digit-reject",
            }

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


def _write_in_candidates(draft_text: str, extra_hypotheses: list[dict[str, Any]] | None) -> list[str]:
    out: list[str] = []
    raw = (draft_text or "").strip()
    if raw:
        out.append(raw)
    for row in extra_hypotheses or []:
        if not isinstance(row, dict):
            continue
        text = str(row.get("value") or row.get("text") or "").strip()
        if text and text not in out:
            out.append(text)
    return out


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
    extra_hypotheses: list[dict[str, Any]] | None = None,
    tau: float,
) -> HandwritingPrediction:
    candidates = _write_in_candidates(draft_text, extra_hypotheses)
    empty = not candidates
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
    if draft_source in {"unavailable", "ink-present", "trocr-nodigit"} and empty:
        return HandwritingPrediction(
            field_id=field_id,
            raw_text="",
            canonical_value=None,
            canonical_id=None,
            confidence=0.0,
            source=draft_source,
            needs_hitl=True,
            hypotheses=hypotheses,
        )

    ranked: list[RankedCandidate] = list(priors)
    accepted: RankedCandidate | None = None
    accepted_score = 0.0
    if kg is not None and candidates:
        for text in candidates:
            ranked = kg.assume(field_id, text, {"ticked_ids": ticked_ids}, top_k=5)
            if not ranked:
                continue
            best = ranked[0]
            hypotheses.append(
                {
                    "value": best.value,
                    "canonical_id": best.canonical_id,
                    "score": best.score,
                    "source": best.reason,
                    "tier": best.tier,
                    "from_draft": text,
                }
            )
            sim = token_similarity(text, best.value)
            score = max(best.score, sim)
            if best.canonical_id and best.tier == 1 and score >= 0.72 and score >= accepted_score:
                accepted = best
                accepted_score = score
    elif kg is not None and not ranked:
        ranked = kg.fuzzy_match_catalogue(draft_text or "", top_k=5)

    if accepted is not None:
        conf = max(draft_conf, accepted_score)
        return HandwritingPrediction(
            field_id=field_id,
            raw_text=draft_text,
            canonical_value=accepted.value,
            canonical_id=accepted.canonical_id,
            confidence=round(min(1.0, conf), 3),
            source=f"{draft_source}+kg",
            needs_hitl=conf < tau,
            hypotheses=hypotheses,
        )

    best = ranked[0] if ranked else None
    if best is not None and not empty:
        # Catalogue hit that is not tier-1: keep raw text, surface the guess, HiTL.
        hypotheses.append(
            {
                "value": best.value,
                "canonical_id": best.canonical_id,
                "score": best.score,
                "source": best.reason,
                "tier": best.tier,
            }
        )
        return HandwritingPrediction(
            field_id=field_id,
            raw_text=draft_text,
            canonical_value=draft_text or best.value,
            canonical_id=None,
            confidence=round(min(1.0, max(best.score, draft_conf) * 0.9), 3),
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

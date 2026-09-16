"""Rescore Block 3 drafts against the frozen KG into an LIS-shaped prediction."""

from __future__ import annotations

from typing import Any

from med_doc.htr.fusion import fuse_handwriting
from med_doc.htr.marks import MAX_PLAUSIBLE_TUBE_COUNT
from med_doc.htr.recognizer import extract_digits
from med_doc.htr.schemas import DocumentHypotheses, DocumentPrediction, HandwritingPrediction
from med_doc.kg.graph import KnowledgeGraph
from med_doc.kg.textutil import normalize_text
from med_doc.rescoring.combinations import (
    CombinationCritic,
    DEFAULT_THRESHOLD,
    CombinationFlag,
    rerun_3a_on_flags,
    select_flags,
)
from med_doc.rescoring.ticks import split_nonverbal_ticks
from med_doc.htr.quality import crop_quality
from med_doc.template_layout import looks_like_htr_garbage

_WRITE_IN_FIELDS = {"others"}
_DATE_FIELDS = {"received_at", "date", "sample_received"}


def _attach_vision_handwriting(hyp: DocumentHypotheses) -> dict[str, HandwritingPrediction]:
    """Copy 3c strings into 3b n-best. Empty/garbage 3b can take vision as the draft."""
    verbal = dict(hyp.verbal)
    vision_hw = (hyp.vision.handwriting if hyp.vision and hyp.vision.source == "vision" else {}) or {}
    for fid, text in vision_hw.items():
        text = (text or "").strip()
        if not text:
            continue
        extra = {"value": text, "score": 0.55, "source": "vision"}
        draft = verbal.get(fid)
        if draft is None:
            verbal[fid] = HandwritingPrediction(
                field_id=fid,
                raw_text=text,
                canonical_value=text,
                confidence=0.55,
                source="vision",
                needs_hitl=True,
                hypotheses=[extra],
            )
            continue
        hyps = list(draft.hypotheses)
        if extra not in hyps:
            hyps.append(extra)
        empty = not (draft.raw_text or "").strip() or looks_like_htr_garbage(draft.raw_text)
        if empty:
            verbal[fid] = draft.model_copy(
                update={
                    "raw_text": text,
                    "canonical_value": text,
                    "confidence": max(float(draft.confidence), 0.55),
                    "source": "vision",
                    "needs_hitl": True,
                    "hypotheses": hyps,
                }
            )
        else:
            verbal[fid] = draft.model_copy(update={"hypotheses": hyps})
    return verbal


def _write_in_blob(verbal: dict[str, HandwritingPrediction]) -> str:
    parts: list[str] = []
    for fid in ("others", "clinical_info"):
        field = verbal.get(fid)
        if field is None:
            continue
        parts.append(field.raw_text or "")
        parts.append(field.canonical_value or "")
        for row in field.hypotheses or []:
            if isinstance(row, dict):
                parts.append(str(row.get("value") or ""))
    return normalize_text(" ".join(parts))


def rank_vision_ticks(
    trusted: list[str],
    vision_ids: list[str],
    verbal: dict[str, HandwritingPrediction],
    kg: KnowledgeGraph,
) -> list[dict[str, Any]]:
    """KG-score 3c-only ticks for HiTL order. Never union into ticked_test_ids."""
    extra = [fid for fid in vision_ids if fid not in set(trusted)]
    implied = kg.implied_tests(trusted)
    blob = _write_in_blob(verbal)
    trusted_tubes = set(kg.calculate_expected_tubes(trusted))
    ranked: list[dict[str, Any]] = []
    for fid in extra:
        score = 0.0
        reasons: list[str] = []
        if fid in implied:
            score += 0.40
            reasons.append("implied_by_profile")
        item = kg.get_item(fid)
        labels = [fid]
        if item is not None:
            labels.extend([item.label, *(item.aliases or [])])
        if blob and any(normalize_text(lab) and normalize_text(lab) in blob for lab in labels):
            score += 0.35
            reasons.append("mentioned_in_writein")
        extra_tubes = set(kg.calculate_expected_tubes(list(trusted) + [fid])) - trusted_tubes
        if not extra_tubes:
            score += 0.20
            reasons.append("tube_consistent")
        ranked.append(
            {
                "field_id": fid,
                "score": round(score, 3),
                "reasons": reasons,
            }
        )
    ranked.sort(key=lambda row: (-float(row["score"]), str(row["field_id"])))
    return ranked


def _vision_tick_disagreements(hyp: DocumentHypotheses, trusted: list[str]) -> list[str]:
    """3c-only ticks stay HiTL. They never join trusted / implied / tubes."""
    vision_ticks = list(hyp.vision.ticked_field_ids if hyp.vision and hyp.vision.source == "vision" else [])
    trusted_set = set(trusted)
    return [fid for fid in vision_ticks if fid not in trusted_set]


def _fuse_field(
    pred: HandwritingPrediction,
    *,
    kg: KnowledgeGraph,
    ticked_ids: list[str],
    expected_tubes: dict[str, int],
) -> HandwritingPrediction:
    return fuse_handwriting(
        pred.field_id,
        pred.raw_text,
        pred.confidence,
        pred.source,
        kg=kg,
        ticked_ids=ticked_ids,
        expected_tubes=expected_tubes,
        extra_hypotheses=pred.hypotheses,
    )


def _parse_observed_count(field: HandwritingPrediction) -> int | None:
    digits = extract_digits(field.canonical_value or "")
    if digits:
        return int(digits)
    return None


def rescore_hypotheses(
    hyp: DocumentHypotheses,
    kg: KnowledgeGraph,
    *,
    critic: CombinationCritic | None = None,
    combo_threshold: float = DEFAULT_THRESHOLD,
    checkbox_crops: dict | None = None,
) -> DocumentPrediction:
    """Turn Block 3 drafts + frozen KG into DocumentPrediction (LIS + HiTL).

    Optional combination critic (small LLM placeholder or scripted flags) ranks
    missing-likely / odd-member ids by semantic confidence. Above ``combo_threshold``
    those crops are re-run through 3a and queued for review. The critic never
    unions a tick into ``ticked_test_ids`` by itself.
    """
    allowed = set(kg.catalogue) | set(hyp.nonverbal)
    trusted0, _uncertain0 = split_nonverbal_ticks(hyp.nonverbal)
    raw_flags: list[CombinationFlag] = []
    if critic is not None:
        raw_flags = critic.flag(list(trusted0), allowed=allowed, kg=kg)
    flags = select_flags(raw_flags, threshold=combo_threshold)
    if flags:
        nonverbal = rerun_3a_on_flags(hyp, flags, checkbox_crops)
        hyp = hyp.model_copy(update={"nonverbal": nonverbal})

    trusted, uncertain = split_nonverbal_ticks(hyp.nonverbal)
    verbal = _attach_vision_handwriting(hyp)
    vision_only_raw = _vision_tick_disagreements(hyp, trusted)
    vision_ranked = rank_vision_ticks(trusted, vision_only_raw, verbal, kg)
    vision_only = [row["field_id"] for row in vision_ranked]

    fused_hw: dict[str, HandwritingPrediction] = {}
    write_in_ids: list[str] = []

    # Write-ins first so accepted catalogue ids can join the tube prior.
    for fid, draft in verbal.items():
        if fid in _WRITE_IN_FIELDS or not (
            fid.startswith("tube_") or fid in _DATE_FIELDS or fid in {"clinical_info", "office_other"}
        ):
            fused = _fuse_field(draft, kg=kg, ticked_ids=trusted, expected_tubes={})
            fused_hw[fid] = fused
            if fid in _WRITE_IN_FIELDS and fused.canonical_id:
                write_in_ids.append(fused.canonical_id)

    ticked = sorted(set(trusted) | set(write_in_ids))
    expected = kg.calculate_expected_tubes(ticked)
    implied = sorted(kg.implied_tests(ticked))

    for fid, draft in verbal.items():
        if fid in fused_hw:
            continue
        fused_hw[fid] = _fuse_field(
            draft, kg=kg, ticked_ids=ticked, expected_tubes=expected
        )

    observed: dict[str, int | None] = {}
    implausible_tubes: list[str] = []
    for fid, field in fused_hw.items():
        if not fid.startswith("tube_"):
            continue
        tube_name = kg.tube_field_to_tube.get(fid, fid)
        count = _parse_observed_count(field)
        if count is not None and count > MAX_PLAUSIBLE_TUBE_COUNT:
            observed[tube_name] = None
            implausible_tubes.append(f"{tube_name}={count}")
            fused_hw[fid] = field.model_copy(update={"needs_hitl": True, "canonical_value": None})
        else:
            observed[tube_name] = count

    observed_for_validate = {k: v for k, v in observed.items() if v is not None}
    report = kg.validate_request(
        ticked,
        observed_tubes=observed_for_validate,
        write_ins=write_in_ids,
    )

    discrepancies = list(report.discrepancies)
    warnings = list(report.warnings)
    combo_hitl: list[str] = []
    for flag in flags:
        pred = hyp.nonverbal.get(flag.field_id)
        committed = (
            flag.kind == "missing_likely"
            and pred is not None
            and pred.is_marked
            and not pred.needs_hitl
        )
        warnings.append(
            f"Combination {flag.kind}: {flag.field_id} "
            f"(confidence={flag.confidence:.2f} {flag.reason})"
        )
        if not committed:
            combo_hitl.append(flag.field_id)
    hitl = list(dict.fromkeys(list(hyp.hitl_fields) + uncertain + vision_only + combo_hitl))
    for note in implausible_tubes:
        warnings.append(f"Implausible tube count ignored: {note}")
    for row in vision_ranked:
        fid = str(row["field_id"])
        reasons = ",".join(row.get("reasons") or []) or "unscored"
        warnings.append(
            f"Vision tick not confirmed by geometry: {fid} (kg_score={row['score']:.2f} {reasons})"
        )

    for fid in uncertain:
        warnings.append(f"Uncertain tick excluded from tube prior: {fid}")
        if fid not in hitl:
            hitl.append(fid)

    for tube_type, exp_count in expected.items():
        obs = observed.get(tube_type)
        field_id = next(
            (fid for fid, name in kg.tube_field_to_tube.items() if name == tube_type),
            f"tube_{tube_type.lower()}",
        )
        if obs is None and exp_count > 0:
            discrepancies.append(
                f"Missing tube observation: expected {exp_count} '{tube_type}' tube(s); crop empty."
            )
            if field_id not in hitl:
                hitl.append(field_id)
        elif obs is not None and obs > exp_count:
            discrepancies.append(
                f"Tube surplus: expected {exp_count} '{tube_type}' tube(s), observed {obs}."
            )
            if field_id not in hitl:
                hitl.append(field_id)
        elif obs is not None and obs != exp_count and field_id not in hitl:
            hitl.append(field_id)

    for fid, field in fused_hw.items():
        if field.needs_hitl and fid not in hitl:
            hitl.append(fid)

    quality = crop_quality(
        hyp=hyp.model_copy(
            update={
                "nonverbal": hyp.nonverbal,
                "verbal": fused_hw,
                "ticked_test_ids": ticked,
                "crop_validate": hyp.crop_validate,
            }
        )
    )
    overall = float(quality["overall_confidence"])

    is_valid = len(discrepancies) == 0

    return DocumentPrediction(
        doc_id=hyp.doc_id,
        is_valid=is_valid,
        checkbox_marks=hyp.nonverbal,
        handwriting_fields=fused_hw,
        ticked_test_ids=ticked,
        implied_tests=implied,
        expected_tubes=expected,
        observed_tubes=observed,
        discrepancies=discrepancies,
        warnings=warnings,
        overall_confidence=round(overall, 3),
        hitl_fields=hitl,
        crop_validate=dict(hyp.crop_validate or {}),
    )


__all__ = ["rescore_hypotheses", "rank_vision_ticks"]

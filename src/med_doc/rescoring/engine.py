"""Rescore Block 3 drafts against the frozen KG into an LIS-shaped prediction."""

from __future__ import annotations

from med_doc.htr.fusion import fuse_handwriting
from med_doc.htr.recognizer import extract_digits
from med_doc.htr.schemas import DocumentHypotheses, DocumentPrediction, HandwritingPrediction
from med_doc.kg.graph import KnowledgeGraph
from med_doc.rescoring.ticks import split_nonverbal_ticks

_WRITE_IN_FIELDS = {"others"}
_DATE_FIELDS = {"received_at", "date", "sample_received"}


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
) -> DocumentPrediction:
    """Turn Block 3 drafts + frozen KG into DocumentPrediction (LIS + HiTL)."""
    trusted, uncertain = split_nonverbal_ticks(hyp.nonverbal)

    fused_hw: dict[str, HandwritingPrediction] = {}
    write_in_ids: list[str] = []

    # Write-ins first so accepted catalogue ids can join the tube prior.
    for fid, draft in hyp.verbal.items():
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

    for fid, draft in hyp.verbal.items():
        if fid in fused_hw:
            continue
        fused_hw[fid] = _fuse_field(
            draft, kg=kg, ticked_ids=ticked, expected_tubes=expected
        )

    observed: dict[str, int | None] = {}
    for fid, field in fused_hw.items():
        if not fid.startswith("tube_"):
            continue
        tube_name = kg.tube_field_to_tube.get(fid, fid)
        observed[tube_name] = _parse_observed_count(field)

    observed_for_validate = {k: v for k, v in observed.items() if v is not None}
    report = kg.validate_request(
        ticked,
        observed_tubes=observed_for_validate,
        write_ins=write_in_ids,
    )

    discrepancies = list(report.discrepancies)
    warnings = list(report.warnings)
    hitl = list(dict.fromkeys(list(hyp.hitl_fields) + uncertain))

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

    confs = [m.confidence for m in hyp.nonverbal.values()] + [h.confidence for h in fused_hw.values()]
    overall = float(sum(confs) / len(confs)) if confs else report.confidence
    overall = min(overall, report.confidence)
    if hitl or discrepancies:
        overall = min(overall, 0.74)

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
    )


__all__ = ["rescore_hypotheses"]

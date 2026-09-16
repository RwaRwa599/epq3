"""Map a committed DocumentPrediction to a generic lab-order payload."""

from __future__ import annotations

from med_doc.htr.quality import crop_quality
from med_doc.htr.schemas import DocumentHypotheses, DocumentPrediction
from med_doc.review.sanity import REGISTRATION_FAILURE, registration_failure_reasons
from med_doc.review.schemas import LabOrder


def order_from_prediction(
    pred: DocumentPrediction,
    hyp: DocumentHypotheses | None = None,
) -> LabOrder:
    received = None
    rec = pred.handwriting_fields.get("received_at")
    if rec is not None and rec.source == "date_parse" and rec.canonical_value:
        received = rec.canonical_value

    others = pred.handwriting_fields.get("others")
    ordered = sorted(set(pred.ticked_test_ids) | set(pred.implied_tests))
    quality = crop_quality(hyp=hyp, pred=pred)
    n_implausible = sum(1 for w in pred.warnings if "Implausible tube count" in w)
    reasons = registration_failure_reasons(
        ticked_test_ids=list(pred.ticked_test_ids),
        ordered_tests=ordered,
        observed_tubes=dict(pred.observed_tubes),
        n_checkbox=int(quality.get("n_checkbox") or len(pred.checkbox_marks) or 0),
        all_tube_crops_empty=bool(quality.get("all_tube_crops_empty")),
        n_implausible_tubes=n_implausible,
    )
    failed = bool(reasons)
    needs_review = bool(pred.hitl_fields) or (not pred.is_valid) or failed
    warnings = list(pred.warnings)
    for line in reasons:
        if line not in warnings:
            warnings.append(line)

    ticked = list(pred.ticked_test_ids)
    implied = list(pred.implied_tests)
    others_id = others.canonical_id if others else None
    others_raw = (others.raw_text or None) if others else None
    if failed:
        n_withheld = len(ordered)
        warnings.append(
            f"{REGISTRATION_FAILURE}: withheld {n_withheld} draft test(s) from LIS order"
        )
        ordered, ticked, implied = [], [], []
        others_id = None
        received = None

    return LabOrder(
        doc_id=pred.doc_id,
        ordered_tests=ordered,
        ticked_test_ids=ticked,
        implied_tests=implied,
        observed_tubes=dict(pred.observed_tubes),
        expected_tubes=dict(pred.expected_tubes),
        received_at=received,
        others_raw=others_raw,
        others_canonical_id=others_id,
        needs_review=needs_review,
        is_valid=False if failed else pred.is_valid,
        discrepancies=list(pred.discrepancies),
        warnings=warnings,
        overall_confidence=float(quality["overall_confidence"]),
        review_reasons=[REGISTRATION_FAILURE] if failed else [],
        registration_failure_suspected=failed,
        crop_retry_rate=float(quality["retry_rate"]),
        crop_hitl_rate=float(quality["hitl_rate"]),
        empty_crop_rate=float(quality["empty_crop_rate"]),
    )

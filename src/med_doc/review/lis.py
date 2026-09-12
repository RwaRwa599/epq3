"""Map a committed DocumentPrediction to a generic lab-order payload."""

from __future__ import annotations

from med_doc.htr.schemas import DocumentPrediction
from med_doc.review.schemas import LabOrder


def order_from_prediction(pred: DocumentPrediction) -> LabOrder:
    received = None
    rec = pred.handwriting_fields.get("received_at")
    if rec is not None and rec.source == "date_parse" and rec.canonical_value:
        received = rec.canonical_value

    others = pred.handwriting_fields.get("others")
    ordered = sorted(set(pred.ticked_test_ids) | set(pred.implied_tests))
    needs_review = bool(pred.hitl_fields) or (not pred.is_valid)
    return LabOrder(
        doc_id=pred.doc_id,
        ordered_tests=ordered,
        ticked_test_ids=list(pred.ticked_test_ids),
        implied_tests=list(pred.implied_tests),
        observed_tubes=dict(pred.observed_tubes),
        expected_tubes=dict(pred.expected_tubes),
        received_at=received,
        others_raw=(others.raw_text or None) if others else None,
        others_canonical_id=others.canonical_id if others else None,
        needs_review=needs_review,
        is_valid=pred.is_valid,
        discrepancies=list(pred.discrepancies),
        warnings=list(pred.warnings),
        overall_confidence=pred.overall_confidence,
    )

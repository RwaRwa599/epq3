"""Apply review patches to Block 3 drafts, then re-run Block 4 rescoring."""

from __future__ import annotations

from med_doc.htr.schemas import DocumentHypotheses, HandwritingPrediction, MarkPrediction
from med_doc.kg.graph import KnowledgeGraph
from med_doc.rescoring.engine import rescore_hypotheses
from med_doc.review.schemas import ReviewPatch


def apply_patches(
    hyp: DocumentHypotheses,
    patches: list[ReviewPatch],
) -> DocumentHypotheses:
    """Return a new hypotheses object with human overrides on drafts.

    Original hypotheses.json on disk is not modified.
    """
    nonverbal = {fid: mark.model_copy() for fid, mark in hyp.nonverbal.items()}
    verbal = {fid: field.model_copy() for fid, field in hyp.verbal.items()}
    hitl = [fid for fid in hyp.hitl_fields]

    for patch in patches:
        fid = patch.field_id
        if patch.action in {"confirm_tick", "reject_tick"}:
            marked = patch.action == "confirm_tick"
            prev = nonverbal.get(fid)
            nonverbal[fid] = MarkPrediction(
                field_id=fid,
                is_marked=marked,
                confidence=1.0,
                ink_density=prev.ink_density if prev else (0.2 if marked else 0.0),
                needs_hitl=False,
                source="hitl",
            )
            hitl = [x for x in hitl if x != fid]
            continue

        raw = (patch.value or "").strip()
        prev_hw = verbal.get(fid)
        hyps = list(prev_hw.hypotheses) if prev_hw else []
        if raw:
            hyps.append({"value": raw, "score": 1.0, "source": "hitl"})
        verbal[fid] = HandwritingPrediction(
            field_id=fid,
            raw_text=raw,
            canonical_value=raw or None,
            canonical_id=fid if fid.startswith("tube_") and raw else None,
            confidence=1.0,
            source="hitl",
            needs_hitl=False,
            hypotheses=hyps,
        )
        hitl = [x for x in hitl if x != fid]

    ticked = [fid for fid, mark in nonverbal.items() if mark.is_marked]
    return hyp.model_copy(
        update={
            "nonverbal": nonverbal,
            "verbal": verbal,
            "ticked_test_ids": ticked,
            "hitl_fields": hitl,
        }
    )


def commit_hypotheses(
    hyp: DocumentHypotheses,
    patches: list[ReviewPatch],
    kg: KnowledgeGraph,
):
    """Patch drafts and re-enter Block 4 constraints."""
    patched = apply_patches(hyp, patches)
    return patched, rescore_hypotheses(patched, kg)

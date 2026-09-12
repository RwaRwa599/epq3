"""Optional LLM ranker. Suggestions never become ticks or tube counts by themselves."""

from __future__ import annotations

from typing import Any

from med_doc.kg.graph import KnowledgeGraph
from med_doc.review.schemas import HitlItem


class LlmRanker:
    """Default: no suggestions. Subclass or inject a callable for experiments."""

    def suggest(
        self,
        item: HitlItem,
        *,
        nbest: list[str],
        kg: KnowledgeGraph | None = None,
    ) -> list[dict[str, Any]]:
        _ = (item, nbest, kg)
        return []


class ScriptedLlmRanker(LlmRanker):
    """Test double: returns a canned string only when it already appears in n-best."""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def suggest(
        self,
        item: HitlItem,
        *,
        nbest: list[str],
        kg: KnowledgeGraph | None = None,
    ) -> list[dict[str, Any]]:
        _ = kg
        text = self.mapping.get(item.field_id)
        if not text or text not in nbest:
            return []
        return [{"value": text, "score": 0.8, "source": "llm"}]


def attach_llm_suggestions(
    queue: list[HitlItem],
    *,
    ranker: LlmRanker | None,
    enabled: bool,
    kg: KnowledgeGraph | None = None,
) -> list[HitlItem]:
    if not enabled or ranker is None:
        return queue
    out: list[HitlItem] = []
    for item in queue:
        if item.kind not in {"write_in", "date"}:
            out.append(item)
            continue
        raw = ranker.suggest(item, nbest=item.nbest, kg=kg)
        filtered = [
            row
            for row in raw
            if isinstance(row, dict) and str(row.get("value") or "") in item.nbest
        ]
        out.append(item.model_copy(update={"llm_suggestions": filtered}))
    return out

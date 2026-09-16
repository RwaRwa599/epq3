"""Optional LLM ranker. Suggestions never become ticks or tube counts by themselves."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

from med_doc.kg.graph import KnowledgeGraph
from med_doc.review.schemas import HitlItem

DEFAULT_OLLAMA = "http://127.0.0.1:11434"
DEFAULT_RANK_MODEL = os.environ.get("MED_DOC_RANK_MODEL", "qwen2.5:7b-instruct")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


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


class OllamaRanker(LlmRanker):
    """Block 5 text Instruct ranker. Reorders write-in/date n-best; never ticks."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        *,
        ticked_ids: list[str] | None = None,
        ticked_labels: list[str] | None = None,
    ):
        self.model = model or DEFAULT_RANK_MODEL
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA).rstrip("/")
        self.ticked_ids = list(ticked_ids or [])
        self.ticked_labels = list(ticked_labels or [])

    def set_ticked_context(self, ids: list[str], labels: list[str] | None = None) -> None:
        self.ticked_ids = list(ids)
        self.ticked_labels = list(labels or ids)

    def _complete(self, prompt: str) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            self.host + "/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        msg = payload.get("message") or {}
        return str(msg.get("content") or payload.get("response") or "")

    def suggest(
        self,
        item: HitlItem,
        *,
        nbest: list[str],
        kg: KnowledgeGraph | None = None,
    ) -> list[dict[str, Any]]:
        if item.kind not in {"write_in", "date"}:
            return []
        if not nbest:
            return []
        labels = self.ticked_labels
        if kg is not None and self.ticked_ids and not labels:
            labels = []
            for fid in self.ticked_ids:
                item_meta = kg.get_item(fid)
                labels.append(item_meta.label if item_meta else fid)
        prompt = (
            "Pick the best transcription from n-best only. JSON: {\"value\":\"...\"}\n"
            f"field={item.field_id} kind={item.kind}\n"
            f"nbest={json.dumps(nbest)}\n"
            f"ticked={json.dumps(labels)}\n"
            "Do not invent a string that is not in nbest. Do not emit ticks or tube counts."
        )
        try:
            raw = self._complete(prompt)
        except Exception:
            return []
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            m = _JSON_RE.search(raw or "")
            if not m:
                return []
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                return []
        if not isinstance(obj, dict):
            return []
        value = str(obj.get("value") or "").strip()
        if value not in nbest:
            return []
        return [{"value": value, "score": 0.7, "source": "ollama"}]


def ollama_ranker_available(host: str | None = None) -> bool:
    from med_doc.htr.vision import ollama_available

    return ollama_available(host)


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

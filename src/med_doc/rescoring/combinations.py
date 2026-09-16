"""Block 4 combination critic: missing-likely and odd-member ticks.

Placeholder until Block 2 has real co-occurrence tables. A small Instruct LLM
(or a scripted test double) scores correlation with the trusted set. High
confidence flags go to review / a second 3a pass — never straight onto LIS.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Literal, Protocol

import numpy as np

from med_doc.htr.marks import classify_mark
from med_doc.htr.schemas import DocumentHypotheses, MarkPrediction
from med_doc.kg.graph import KnowledgeGraph

DEFAULT_OLLAMA = "http://127.0.0.1:11434"
DEFAULT_COMBO_MODEL = os.environ.get("MED_DOC_COMBO_MODEL", "llama3.2:3b-instruct")
DEFAULT_THRESHOLD = 0.55
MAX_COMBO_FLAGS = 16
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

FlagKind = Literal["missing_likely", "odd_member"]


class CombinationFlag:
    __slots__ = ("field_id", "kind", "confidence", "reason")

    def __init__(
        self,
        field_id: str,
        kind: FlagKind,
        confidence: float,
        reason: str = "",
    ):
        self.field_id = field_id
        self.kind = kind
        self.confidence = float(max(0.0, min(1.0, confidence)))
        self.reason = reason

    def as_dict(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "kind": self.kind,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
        }


class CombinationCritic(Protocol):
    def flag(
        self,
        trusted: list[str],
        *,
        allowed: set[str],
        kg: KnowledgeGraph | None = None,
    ) -> list[CombinationFlag]: ...


class NullCombinationCritic:
    def flag(self, trusted, *, allowed, kg=None) -> list[CombinationFlag]:
        _ = (trusted, allowed, kg)
        return []


class ScriptedCombinationCritic:
    """Tests: canned flags. Still filtered by allowed ids in ``select_flags``."""

    def __init__(self, flags: list[CombinationFlag | dict[str, Any]]):
        self._flags = [
            f if isinstance(f, CombinationFlag) else CombinationFlag(**f) for f in flags
        ]

    def flag(self, trusted, *, allowed, kg=None) -> list[CombinationFlag]:
        _ = (trusted, kg)
        return [f for f in self._flags if f.field_id in allowed]


class KgBundleCritic:
    """Weak stand-in from profile_bundles until co-occurrence tables exist.

    Missing-likely = profile components not ticked. Odd-member = a ticked test
    that is already a component of a ticked profile (redundant).
    """

    def flag(self, trusted, *, allowed, kg=None) -> list[CombinationFlag]:
        if kg is None:
            return []
        trusted_set = set(trusted)
        out: list[CombinationFlag] = []
        implied = kg.implied_tests(list(trusted))
        for fid in sorted(implied):
            if fid in trusted_set or fid not in allowed:
                continue
            out.append(
                CombinationFlag(
                    fid,
                    "missing_likely",
                    0.62,
                    reason="profile_component_not_ticked",
                )
            )
        for fid in trusted:
            if fid in implied and fid not in kg.profile_bundles and fid in allowed:
                out.append(
                    CombinationFlag(
                        fid,
                        "odd_member",
                        0.60,
                        reason="redundant_with_ticked_profile",
                    )
                )
        return out


class OllamaCombinationCritic:
    """Placeholder for frozen combination tables: small local Instruct model."""

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or DEFAULT_COMBO_MODEL
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA).rstrip("/")

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

    def flag(self, trusted, *, allowed, kg=None) -> list[CombinationFlag]:
        labels = []
        if kg is not None:
            for fid in trusted:
                item = kg.get_item(fid)
                labels.append(item.label if item else fid)
        prompt = (
            "Lab request checkboxes. JSON only.\n"
            '{"missing_likely":[{"field_id":"","confidence":0.0,"reason":""}],'
            '"odd_member":[{"field_id":"","confidence":0.0,"reason":""}]}\n'
            "missing_likely: not ticked but highly correlated with the trusted set "
            "(high correlation = high confidence).\n"
            "odd_member: ticked but does not belong with this combination.\n"
            f"trusted={json.dumps(list(trusted))}\n"
            f"trusted_labels={json.dumps(labels)}\n"
            "field_id must be in: "
            + ",".join(sorted(allowed)[:200])
            + "\nDo not add ticks to the order. Scores are 0-1."
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
        out: list[CombinationFlag] = []
        for kind in ("missing_likely", "odd_member"):
            for row in obj.get(kind) or []:
                if not isinstance(row, dict):
                    continue
                fid = str(row.get("field_id") or "").strip()
                if fid not in allowed:
                    continue
                if kind == "missing_likely" and fid in set(trusted):
                    continue
                if kind == "odd_member" and fid not in set(trusted):
                    continue
                try:
                    conf = float(row.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    conf = 0.0
                out.append(
                    CombinationFlag(
                        fid,
                        kind,  # type: ignore[arg-type]
                        conf,
                        reason=str(row.get("reason") or "llm_placeholder")[:120],
                    )
                )
        return out


def select_flags(
    flags: list[CombinationFlag],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    cap: int = MAX_COMBO_FLAGS,
) -> list[CombinationFlag]:
    """High semantic confidence → review. Below threshold is ignored."""
    kept = [f for f in flags if f.confidence >= threshold]
    kept.sort(key=lambda f: (-f.confidence, f.kind, f.field_id))
    return kept[:cap]


def rerun_3a_on_flags(
    hyp: DocumentHypotheses,
    flags: list[CombinationFlag],
    crops: dict[str, np.ndarray | None] | None,
) -> dict[str, MarkPrediction]:
    """Second 3a pass on flagged checkbox crops only. LLM never writes LIS ticks."""
    nonverbal = dict(hyp.nonverbal)
    crops = crops or {}
    for flag in flags:
        crop = crops.get(flag.field_id)
        if crop is None or getattr(crop, "size", 0) == 0:
            continue
        pred = classify_mark(np.asarray(crop), flag.field_id)
        if flag.kind == "missing_likely":
            nonverbal[flag.field_id] = pred
        elif flag.kind == "odd_member":
            if pred.is_marked:
                nonverbal[flag.field_id] = pred.model_copy(update={"needs_hitl": True})
            else:
                nonverbal[flag.field_id] = pred
    return nonverbal


def make_critic(backend: str, *, model: str | None = None) -> CombinationCritic:
    key = (backend or "off").strip().lower()
    if key in {"off", "none", ""}:
        return NullCombinationCritic()
    if key in {"kg", "bundles"}:
        return KgBundleCritic()
    if key in {"ollama", "llm", "instruct"}:
        return OllamaCombinationCritic(model=model)
    raise ValueError(f"unknown combo_backend: {backend}")

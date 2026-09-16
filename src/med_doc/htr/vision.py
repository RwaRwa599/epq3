"""Block 3c: independent local vision-language draft (Ollama). Not tick authority."""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Protocol

import cv2
import numpy as np

from med_doc.htr.ingest import Block1Document
from med_doc.htr.schemas import VisionDraft

DEFAULT_OLLAMA = "http://127.0.0.1:11434"
DEFAULT_MODEL = os.environ.get("MED_DOC_VISION_MODEL", "qwen2.5vl:7b")
MAX_HW_CROPS = 12
MAX_DISAGREEMENT_CROPS = 16
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class VisionClient(Protocol):
    model: str

    def complete(self, prompt: str, images_jpeg: list[bytes]) -> str: ...


class ScriptedVisionClient:
    """Tests: return a canned JSON string."""

    def __init__(self, payload: dict[str, Any], model: str = "scripted"):
        self.payload = payload
        self.model = model
        self.calls: list[tuple[str, list[bytes]]] = []

    def complete(self, prompt: str, images_jpeg: list[bytes]) -> str:
        self.calls.append((prompt, list(images_jpeg)))
        return json.dumps(self.payload)


def ollama_available(host: str | None = None) -> bool:
    url = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA).rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as resp:
            return int(getattr(resp, "status", 200) or 200) < 400
    except Exception:
        return False


class OllamaVisionClient:
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or DEFAULT_MODEL
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA).rstrip("/")

    def complete(self, prompt: str, images_jpeg: list[bytes]) -> str:
        b64 = [base64.b64encode(b).decode("ascii") for b in images_jpeg]
        body = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": b64,
                }
            ],
        }
        req = urllib.request.Request(
            self.host + "/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        msg = payload.get("message") or {}
        return str(msg.get("content") or payload.get("response") or "")


def _jpeg(rgb: np.ndarray, *, max_side: int = 1024) -> bytes:
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale < 0.999:
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return bytes(buf)


def parse_vision_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        m = _JSON_RE.search(raw)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}


def whitelist_vision_payload(
    payload: dict[str, Any],
    *,
    allowed_ticks: set[str],
    allowed_hw: set[str],
) -> tuple[list[str], dict[str, str]]:
    ticks: list[str] = []
    for fid in payload.get("ticked_field_ids") or []:
        key = str(fid).strip()
        if key in allowed_ticks and key not in ticks:
            ticks.append(key)
    hw: dict[str, str] = {}
    blob = payload.get("handwriting") or {}
    if isinstance(blob, dict):
        for fid, val in blob.items():
            key = str(fid).strip()
            if key not in allowed_hw:
                continue
            text = str(val or "").strip()
            if text:
                hw[key] = text[:240]
    return ticks, hw


def _catalogue_ids(doc: Block1Document) -> tuple[set[str], set[str]]:
    fields = doc.fields or {}
    ticks = set((fields.get("checkboxes") or {}).keys()) or set(doc.checkbox_crops)
    hw = set((fields.get("handwriting") or {}).keys()) or set(doc.handwriting_crops)
    return ticks, hw


def disagreement_tick_ids(geometry_ids: set[str], vision_ids: set[str]) -> list[str]:
    """3c tick not in 3a, or 3a tick 3c omitted. Not the full 138-box grid."""
    return sorted((vision_ids - geometry_ids) | (geometry_ids - vision_ids))


def disagreement_crop_ids(
    doc: Block1Document,
    geometry_ids: set[str],
    vision_ids: set[str],
    *,
    cap: int = MAX_DISAGREEMENT_CROPS,
) -> list[str]:
    """Checkbox crop ids for a second 3c pass, capped so the VLM is not classifying the grid."""
    out: list[str] = []
    for fid in disagreement_tick_ids(geometry_ids, vision_ids):
        crop = doc.checkbox_crops.get(fid)
        if crop is None or getattr(crop, "size", 0) == 0:
            continue
        out.append(fid)
        if len(out) >= cap:
            break
    return out


def _page_hw_prompt(tick_ids: list[str], hw_ids: list[str]) -> str:
    return (
        "Lab request form. Reply with JSON only, no prose.\n"
        '{"ticked_field_ids":["field_id"],"handwriting":{"others":"","clinical_info":"","tube_edta":""}}\n'
        "ticked_field_ids must be a subset of: "
        + ",".join(tick_ids)
        + "\n"
        "handwriting keys must be a subset of: "
        + ",".join(hw_ids)
        + "\n"
        "Empty boxes stay unmarked. Invented ids are forbidden. "
        "Images: 1) full page, then handwriting crops in that key order."
    )


def _disagreement_prompt(field_ids: list[str]) -> str:
    return (
        "Checkbox crops that disagree with geometry ticks. JSON only.\n"
        '{"ticked_field_ids":["field_id"]}\n'
        "Only these field ids, in image order: "
        + ",".join(field_ids)
        + "\n"
        "Include an id only if that crop is ticked. Do not invent ids."
    )


def _encode_page_and_hw(doc: Block1Document, allowed_hw: set[str]) -> tuple[list[bytes], list[str]]:
    images: list[bytes] = []
    page = None
    can = doc.doc_dir / "canonical.png"
    if can.is_file():
        from med_doc.htr.ingest import load_rgb

        page = load_rgb(can)
    if page is not None:
        images.append(_jpeg(page, max_side=1280))
    hw_order = [fid for fid in sorted(allowed_hw) if doc.handwriting_crops.get(fid) is not None]
    for fid in hw_order[:MAX_HW_CROPS]:
        crop = doc.handwriting_crops.get(fid)
        if crop is None:
            continue
        images.append(_jpeg(np.asarray(crop), max_side=512))
    return images, hw_order


def run_vision_draft(
    doc: Block1Document,
    *,
    client: VisionClient | None = None,
    enabled: bool = False,
    model: str | None = None,
    geometry_ticks: list[str] | None = None,
    max_disagreement_crops: int = MAX_DISAGREEMENT_CROPS,
) -> VisionDraft:
    """Independent 3c draft. Off / missing Ollama → source=off|unavailable, not an error.

    Pass 1: canonical page + handwriting crops (not the checkbox grid).
    Pass 2 (optional): only 3a/3c disagreement checkbox crops, capped.
    """
    if not enabled and client is None:
        return VisionDraft(source="off", notes="vision_backend=off")
    if client is None:
        if not ollama_available():
            return VisionDraft(source="unavailable", notes="ollama not reachable on 127.0.0.1:11434")
        client = OllamaVisionClient(model=model)
    elif model:
        try:
            client.model = model  # type: ignore[misc]
        except Exception:
            pass
    allowed_ticks, allowed_hw = _catalogue_ids(doc)
    if not allowed_ticks and not allowed_hw:
        return VisionDraft(source=getattr(client, "model", "vision"), notes="no crops")

    images, hw_order = _encode_page_and_hw(doc, allowed_hw)
    prompt = _page_hw_prompt(sorted(allowed_ticks), hw_order or sorted(allowed_hw))
    try:
        raw = client.complete(prompt, images)
    except Exception as exc:
        return VisionDraft(
            source="error",
            model=getattr(client, "model", ""),
            notes=str(exc)[:200],
        )
    payload = parse_vision_json(raw)
    ticks, hw = whitelist_vision_payload(
        payload, allowed_ticks=allowed_ticks, allowed_hw=allowed_hw
    )

    notes = "independent of 3a/3b; Block 4 ranks; VLM ticks are not trusted"
    geom = set(geometry_ticks or [])
    disagree = disagreement_crop_ids(
        doc, geom, set(ticks), cap=max_disagreement_crops
    )
    if disagree:
        crop_images: list[bytes] = []
        for fid in disagree:
            crop = doc.checkbox_crops.get(fid)
            if crop is None:
                continue
            crop_images.append(_jpeg(np.asarray(crop), max_side=256))
        if crop_images:
            try:
                raw2 = client.complete(_disagreement_prompt(disagree), crop_images)
                payload2 = parse_vision_json(raw2)
                refine, _ = whitelist_vision_payload(
                    payload2, allowed_ticks=set(disagree), allowed_hw=set()
                )
                keep_agreed = [fid for fid in ticks if fid not in set(disagree)]
                ticks = list(dict.fromkeys(keep_agreed + refine))
                notes += f"; disagreement pass n={len(disagree)}"
            except Exception as exc:
                notes += f"; disagreement pass failed: {str(exc)[:80]}"

    return VisionDraft(
        source="vision",
        model=getattr(client, "model", ""),
        ticked_field_ids=ticks,
        handwriting=hw,
        needs_hitl=True,
        notes=notes,
    )

"""Ingest a Block 1 (or Block 2-superset) ZIP/folder of normalized crops."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np


def load_rgb(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def unzip_or_dir(input_source: str | Path) -> tuple[Path, bool]:
    input_path = Path(input_source)
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        temp_in = Path(tempfile.mkdtemp(prefix="block3_in_"))
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(temp_in)
        return temp_in, True
    if input_path.is_dir():
        return input_path, False
    raise ValueError(f"Invalid Block 1 input source: {input_source}")


@dataclass
class Block1Document:
    """One document from a Block 1 ZIP (`docs/<id>/`)."""

    doc_id: str
    doc_dir: Path
    metadata: dict[str, Any]
    checkbox_crops: dict[str, np.ndarray | None] = field(default_factory=dict)
    handwriting_crops: dict[str, np.ndarray | None] = field(default_factory=dict)
    detected_marks: dict[str, Any] = field(default_factory=dict)

    @property
    def fields(self) -> dict[str, Any]:
        return self.metadata.get("fields") or {}


def load_block1_document(doc_dir: str | Path) -> Block1Document:
    """Load one `docs/<id>/` folder (metadata.json + crops)."""
    doc_dir = Path(doc_dir)
    meta_path = doc_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {doc_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fields = meta.get("fields") or {}
    cb_meta = fields.get("checkboxes") or {}
    hw_meta = fields.get("handwriting") or {}
    detected = meta.get("detected_marks") or {}
    checkbox_crops = {
        fid: load_rgb(doc_dir / ((info or {}).get("crop_path") or f"crops/checkboxes/{fid}.png"))
        for fid, info in cb_meta.items()
    }
    handwriting_crops = {
        fid: load_rgb(doc_dir / ((info or {}).get("crop_path") or f"crops/handwriting/{fid}.png"))
        for fid, info in hw_meta.items()
    }
    return Block1Document(
        doc_id=str(meta.get("doc_id") or doc_dir.name),
        doc_dir=doc_dir,
        metadata=meta,
        checkbox_crops=checkbox_crops,
        handwriting_crops=handwriting_crops,
        detected_marks=detected,
    )


def iter_block1_documents(base_dir: Path) -> Iterator[Block1Document]:
    """Yield documents listed in `manifest.json`, else every `docs/*` folder."""
    manifest_file = base_dir / "manifest.json"
    entries: list[dict[str, Any]]
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        entries = list(manifest.get("documents") or [])
    else:
        entries = [{"doc_id": p.name} for p in sorted((base_dir / "docs").glob("*")) if p.is_dir()]

    for entry in entries:
        if entry.get("status") == "error":
            continue
        doc_id = str(entry.get("doc_id") or "")
        if not doc_id:
            continue
        doc_dir = base_dir / "docs" / doc_id
        if not (doc_dir / "metadata.json").exists():
            continue
        yield load_block1_document(doc_dir)


def cleanup_temp(path: Path, is_temp: bool) -> None:
    if is_temp:
        shutil.rmtree(path, ignore_errors=True)

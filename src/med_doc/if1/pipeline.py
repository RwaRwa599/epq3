"""if1 runner: same Blocks 1–5 driver with ``if1=True``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from med_doc.kg.graph import KnowledgeGraph
from med_doc.pipeline import run_blocks_1_to_5
from med_doc.review.schemas import OutputMode


def run_if1(
    inputs: str | Path | list,
    *,
    output_dir: str | Path | None = None,
    kg: KnowledgeGraph | None = None,
    output_mode: OutputMode = "user",
    **kwargs: Any,
) -> dict[str, Any]:
    """Working pipeline with if1block1/3/4 embedded. Block 5 writes ``order.json``."""
    kwargs.setdefault("htr_mode", "nonverbal")
    kwargs.setdefault("mark_backend", "geometry")
    kwargs.setdefault("vision_backend", "off")
    return run_blocks_1_to_5(
        inputs,
        output_dir=output_dir,
        kg=kg,
        output_mode=output_mode,
        if1=True,
        **kwargs,
    )

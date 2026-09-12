"""Collect image paths or arrays for Block 1 / 1a batch runners."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Any, Sequence

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


def _unique_id(stem: str, used: dict[str, int]) -> str:
    n = used.get(stem, 0) + 1
    used[stem] = n
    return stem if n == 1 else f"{stem}_{n}"


def collect_image_inputs(
    inputs: Sequence[str | Path | Any] | str | Path,
) -> list[tuple[str, Any]]:
    """Return ``(doc_id, path_or_array)`` pairs from a dir, ZIP, file, or list."""
    items: list[tuple[str, Any]] = []
    used: dict[str, int] = {}

    if isinstance(inputs, (str, Path)):
        inp_path = Path(inputs)
        if inp_path.is_file() and inp_path.suffix.lower() == ".zip":
            temp_in = Path(tempfile.mkdtemp(prefix="raw_batch_"))
            with zipfile.ZipFile(inp_path, "r") as zf:
                zf.extractall(temp_in)
            for p in sorted(temp_in.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                    items.append((_unique_id(p.stem, used), p))
        elif inp_path.is_dir():
            for p in sorted(inp_path.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                    items.append((_unique_id(p.stem, used), p))
        elif inp_path.is_file() and inp_path.suffix.lower() in IMAGE_SUFFIXES:
            items.append((_unique_id(inp_path.stem, used), inp_path))
        else:
            raise ValueError(f"No valid image files found in input: {inputs}")
    else:
        for idx, item in enumerate(inputs):
            if isinstance(item, (str, Path)):
                p = Path(item)
                if p.is_dir():
                    items.extend(collect_image_inputs(p))
                    continue
                if p.is_file() and p.suffix.lower() == ".zip":
                    items.extend(collect_image_inputs(p))
                    continue
                items.append((_unique_id(p.stem, used), p))
            else:
                items.append((_unique_id(f"doc_{idx + 1:03d}", used), item))

    if not items:
        raise ValueError(f"No valid image files found in input: {inputs}")
    return items

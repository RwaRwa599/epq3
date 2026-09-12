"""Colab bootstrap — BOOTSTRAP_V5 (always refresh zipball; drop cached med_doc)."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


def _content() -> Path:
    return Path("/content") if Path("/content").is_dir() else Path.cwd()


DEFAULT_ZIP_URL = "https://codeload.github.com/RwaRwa599/epq3/zip/refs/heads/block1"


def _purge_med_doc() -> None:
    for name in list(sys.modules):
        if name == "med_doc" or name.startswith("med_doc."):
            del sys.modules[name]
    importlib.invalidate_caches()


def bootstrap_med_doc(*, content: Path | None = None, url: str | None = None) -> Path:
    """Download epq3@block1 zip every time, then put src/ on sys.path."""
    content = content or _content()
    zip_path = content / "epq3-block1.zip"
    url = url or DEFAULT_ZIP_URL
    print("Downloading", url)
    urllib.request.urlretrieve(url, zip_path)

    extract_dir = content / "_epq3_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    found = list(extract_dir.glob("*/src/med_doc/__init__.py"))
    if not found:
        raise RuntimeError(f"zip had no src/med_doc: {list(extract_dir.iterdir())}")
    unpacked = found[0].parents[2]
    dest = content / "epq3"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(unpacked), str(dest))
    shutil.rmtree(extract_dir, ignore_errors=True)
    zip_path.unlink(missing_ok=True)
    _activate(dest)
    return dest


def _activate(root: Path) -> None:
    src = str((root / "src").resolve())
    while src in sys.path:
        sys.path.remove(src)
    sys.path.insert(0, src)
    os.chdir(root)
    _purge_med_doc()
    print("BOOTSTRAP_V5")
    print("repo:", root)
    print("cwd:", os.getcwd())
    print("sys.path[0]:", sys.path[0])


def guard_med_doc() -> None:
    content = _content()
    hits = list(content.glob("epq3/src/med_doc/__init__.py"))
    hits += list(content.glob("*/src/med_doc/__init__.py"))
    if not hits:
        bootstrap_med_doc()
        return
    src = str(hits[0].parents[1].resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    os.chdir(hits[0].parents[2])

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

KG_DIR = PACKAGE_ROOT / "kg"
DEFAULT_KG = KG_DIR / "lab_request_v1_kg.json"
V0_KG = KG_DIR / "lab_request_v0_kg.json"

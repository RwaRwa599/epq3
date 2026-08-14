from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
ROOT = SRC_DIR.parent

TEMPLATES_DIR = ROOT / "templates"
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
SYNTHETIC_DIR = SAMPLES_DIR / "synthetic"
PRIVATE_SAMPLES_DIR = SAMPLES_DIR / "private"
OUTPUTS_DIR = ROOT / "outputs"

DEFAULT_TEMPLATE = TEMPLATES_DIR / "lab_request_canonical.json"
V1_TEMPLATE = TEMPLATES_DIR / "lab_request_v1_canonical.json"

CANONICAL_SIZE = (2048, 1720)  # 2× the digital v0 blank (1024×860)
V1_CANONICAL_SIZE = (2048, 1754)  # 2× clinic print v1 (1024×877)


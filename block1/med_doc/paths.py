"""Path definitions and canonical dimensions for Block 1."""

from __future__ import annotations

from pathlib import Path

# Package and root paths
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PACKAGE_ROOT / "templates"
SAMPLES_DIR = PACKAGE_ROOT / "samples"
OUTPUTS_DIR = PACKAGE_ROOT / "outputs"

DEFAULT_TEMPLATE = TEMPLATES_DIR / "lab_request_canonical.json"
V1_TEMPLATE = TEMPLATES_DIR / "lab_request_v1_canonical.json"

# Dimensions
CANONICAL_SIZE = (2048, 1720)      # v0 digital canonical (width, height)
V1_CANONICAL_SIZE = (2048, 1754)   # v1 clinic print canonical (width, height)
CANONICAL_WIDTH = 2048
CANONICAL_HEIGHT = 1720
V1_CANONICAL_WIDTH = 2048
V1_CANONICAL_HEIGHT = 1754

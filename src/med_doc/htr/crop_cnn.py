"""Optional 32×32 residual CNN for ticks — not used unless logreg plateaus.

Do not import this from ``classify_mark``. A CNN is the next step after a labeled
clinic crop set still has high FP/FN with ``train_mark_logreg``. Weights would live
in gitignored ``data/labels/clinic/tick_cnn.pt``.
"""

from __future__ import annotations

from typing import Any


def cnn_available() -> bool:
    return False


def classify_residual_crop(_crop: Any, field_id: str) -> None:
    raise RuntimeError(
        "Tick CNN is not enabled. Label Block 1 crops and refit the logistic "
        f"(python -m med_doc.eval) before considering a CNN for {field_id}."
    )

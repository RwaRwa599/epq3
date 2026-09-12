"""Synthetic + clinic crop features → refit mark logistic. No PHI in git."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from med_doc.eval.photoreal import distort_sheet
from med_doc.htr import marks as marks_mod
from med_doc.htr.marks import (
    classify_mark,
    mark_features,
    save_logreg_weights,
    train_mark_logreg,
)


def _empty(size: int = 32) -> np.ndarray:
    img = np.full((size, size, 3), 245, dtype=np.uint8)
    img[0:2, :] = 20
    img[-2:, :] = 20
    img[:, 0:2] = 20
    img[:, -2:] = 20
    return img


def _slash(size: int = 32) -> np.ndarray:
    img = _empty(size)
    for i in range(6, size - 6):
        img[i, i] = 15
        if i + 1 < size - 6:
            img[i, i + 1] = 15
    return img


def _v_tick(size: int = 32) -> np.ndarray:
    img = _empty(size)
    cv2.line(img, (8, 14), (14, 24), (18, 18, 18), 2)
    cv2.line(img, (14, 24), (24, 8), (18, 18, 18), 2)
    return img


def _fill(size: int = 32) -> np.ndarray:
    img = _empty(size)
    img[6:-6, 6:-6] = 25
    return img


def _corner(size: int = 32) -> np.ndarray:
    img = np.full((size, size, 3), 245, dtype=np.uint8)
    img[0:4, :] = 20
    img[:, 0:4] = 20
    return img


def _label_strip(size: int = 32) -> np.ndarray:
    img = np.full((size, size, 3), 250, dtype=np.uint8)
    img[10:22, 2:-2] = 30
    return img


def synthetic_xy(*, empty_repeats: int = 8, distort_seeds: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heavy empty mix (printed rings, corners, labels) vs slash/V/fill ticks."""
    empties = [_empty(), _corner(), _label_strip()]
    ticks = [_slash(), _v_tick(), _fill()]
    Xs: list[np.ndarray] = []
    ys: list[float] = []
    ws: list[float] = []
    for seed in range(max(distort_seeds, 1)):
        for img in empties:
            crop = distort_sheet(img, seed=seed, strength=0.3) if seed else img
            Xs.append(mark_features(crop))
            ys.append(0.0)
            ws.append(float(empty_repeats))
        for img in ticks:
            crop = distort_sheet(img, seed=seed + 17, strength=0.3) if seed else img
            Xs.append(mark_features(crop))
            ys.append(1.0)
            ws.append(1.0)
    return np.stack(Xs), np.asarray(ys, dtype=np.float64), np.asarray(ws, dtype=np.float64)


def load_crop_dir(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ``{doc}/empty/*.png`` and ``{doc}/tick/*.png`` (clinic, gitignored)."""
    root = Path(root)
    Xs: list[np.ndarray] = []
    ys: list[float] = []
    ws: list[float] = []
    for png in sorted(root.glob("**/empty/*.png")):
        bgr = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        Xs.append(mark_features(rgb))
        ys.append(0.0)
        ws.append(8.0)
    for png in sorted(root.glob("**/tick/*.png")):
        bgr = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        Xs.append(mark_features(rgb))
        ys.append(1.0)
        ws.append(1.0)
    if not Xs:
        return (
            np.zeros((0, 8), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    return np.stack(Xs), np.asarray(ys), np.asarray(ws)


def fit_mark_logreg(
    *,
    crop_dir: str | Path | None = None,
    empty_repeats: int = 8,
    steps: int = 800,
) -> tuple[np.ndarray, float, dict[str, int]]:
    X_s, y_s, w_s = synthetic_xy(empty_repeats=empty_repeats)
    n_clinic_empty = n_clinic_tick = 0
    if crop_dir is not None and Path(crop_dir).is_dir():
        X_c, y_c, w_c = load_crop_dir(crop_dir)
        n_clinic_empty = int((y_c == 0).sum())
        n_clinic_tick = int((y_c == 1).sum())
        if len(y_c):
            X_s = np.concatenate([X_s, X_c], axis=0)
            y_s = np.concatenate([y_s, y_c], axis=0)
            w_s = np.concatenate([w_s, w_c], axis=0)
    w, b = train_mark_logreg(X_s, y_s, steps=steps, sample_weight=w_s)
    stats = {
        "n_rows": int(len(y_s)),
        "n_empty": int((y_s == 0).sum()),
        "n_tick": int((y_s == 1).sum()),
        "n_clinic_empty": n_clinic_empty,
        "n_clinic_tick": n_clinic_tick,
    }
    return w, b, stats


def train_and_save(
    dest: str | Path,
    *,
    crop_dir: str | Path | None = None,
    empty_repeats: int = 8,
) -> dict[str, object]:
    w, b, stats = fit_mark_logreg(crop_dir=crop_dir, empty_repeats=empty_repeats)
    path = save_logreg_weights(dest, w, b)
    marks_mod.LOGREG_W = np.asarray(w, dtype=np.float64)
    marks_mod.LOGREG_B = float(b)
    stats["path"] = str(path)
    stats["w"] = [float(x) for x in w.tolist()]
    stats["b"] = float(b)
    # Smoke: synthetic empty vs slash still separate after the refit.
    stats["empty_marked"] = bool(classify_mark(_empty(), "alt").is_marked)
    stats["slash_marked"] = bool(classify_mark(_slash(), "cbc").is_marked)
    return stats

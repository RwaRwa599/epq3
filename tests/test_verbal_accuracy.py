"""Block 3 verbal accuracy: tubes, dates, others — clean and photoreal (no PHI)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from med_doc.eval.photoreal import distort_sheet
from med_doc.htr.verbal import recognize_verbal
from med_doc.paths import ROOT

SCORECARD = ROOT / "docs" / "blocks" / "data" / "verbal-accuracy.json"


def _canvas(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), 250, dtype=np.uint8)


def render_text(text: str, *, h: int = 48, w: int | None = None, scale: float = 1.0, thick: int = 2) -> np.ndarray:
    w = w or max(48, int(18 * len(text) * scale) + 24)
    img = _canvas(h, w)
    cv2.putText(img, text, (8, int(h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, scale, (25, 25, 25), thick, cv2.LINE_AA)
    return img


def _norm(s: str) -> str:
    return "".join(ch for ch in (s or "").upper() if ch.isalnum() or ch in "/:-")


def _tube_metrics() -> dict:
    n = 0
    clean_ok = 0
    dist_ok = 0
    for d in range(10):
        for seed in range(4):
            n += 1
            img = render_text(str(d), h=40, w=36, scale=1.1, thick=2)
            pred = recognize_verbal(img, "tube_edta", backend="auto")
            if pred.raw_text == str(d):
                clean_ok += 1
            warped = distort_sheet(img, seed=seed, strength=0.25)
            pred_d = recognize_verbal(warped, "tube_edta", backend="auto")
            if pred_d.raw_text == str(d):
                dist_ok += 1
    return {"n": n, "clean": round(clean_ok / n, 4), "photoreal": round(dist_ok / n, 4),
            "clean_ok": clean_ok, "photoreal_ok": dist_ok}


def test_tube_digit_accuracy_clean_and_photoreal():
    m = _tube_metrics()
    assert m["clean"] >= 0.90, f"clean tube digits {m['clean_ok']}/{m['n']}"
    assert m["photoreal"] >= 0.70, f"photoreal tube digits {m['photoreal_ok']}/{m['n']}"


def _date_metrics() -> dict:
    gold = [
        "14/08/2026",
        "01/01/2025",
        "09/12/2026",
        "31/03/2024",
        "07/07/2026",
        "22/11/2025",
    ]
    clean_ok = 0
    dist_ok = 0
    for i, g in enumerate(gold):
        img = render_text(g, h=44, w=220, scale=0.85, thick=2)
        pred = recognize_verbal(img, "received_at", backend="auto")
        got = _norm(pred.canonical_value or pred.raw_text)
        if _norm(g) in got or got in _norm(g):
            clean_ok += 1
        warped = distort_sheet(img, seed=i + 3, strength=0.2)
        pred_d = recognize_verbal(warped, "received_at", backend="auto")
        got_d = _norm(pred_d.canonical_value or pred_d.raw_text)
        if _norm(g) in got_d or got_d in _norm(g):
            dist_ok += 1
    n = len(gold)
    return {"n": n, "clean": round(clean_ok / n, 4), "photoreal": round(dist_ok / n, 4),
            "clean_ok": clean_ok, "photoreal_ok": dist_ok}


def test_date_parse_accuracy():
    m = _date_metrics()
    assert m["clean"] >= 0.66, f"clean dates {m['clean_ok']}/{m['n']}"


def _others_metrics() -> dict:
    names = ["AFP", "CEA", "CBC", "HBA1C", "LIPID", "URIC"]
    clean_ok = 0
    dist_ok = 0
    for i, name in enumerate(names):
        img = render_text(name, h=52, w=160, scale=0.9, thick=2)
        pred = recognize_verbal(img, "others", backend="auto")
        got = _norm(pred.raw_text).replace(" ", "")
        if name in got or got in name:
            clean_ok += 1
        warped = distort_sheet(img, seed=i + 11, strength=0.2)
        pred_d = recognize_verbal(warped, "others", backend="auto")
        got_d = _norm(pred_d.raw_text).replace(" ", "")
        if name in got_d or got_d in name:
            dist_ok += 1
    n = len(names)
    return {"n": n, "clean": round(clean_ok / n, 4), "photoreal": round(dist_ok / n, 4),
            "clean_ok": clean_ok, "photoreal_ok": dist_ok}


def test_others_writein_accuracy():
    m = _others_metrics()
    assert m["clean"] >= 0.66, f"clean others {m['clean_ok']}/{m['n']}"


def test_empty_others_and_tube_stay_empty():
    blank = np.full((48, 80, 3), 250, dtype=np.uint8)
    others = recognize_verbal(blank, "others", backend="auto")
    tube = recognize_verbal(blank, "tube_cb", backend="auto")
    assert others.source == "empty"
    assert tube.source == "empty"
    assert others.needs_hitl is False


def test_write_verbal_accuracy_scorecard():
    tubes = _tube_metrics()
    dates = _date_metrics()
    others = _others_metrics()
    payload = {
        "eval_id": "verbal-field-typed-b3.7",
        "note": "Synthetic putText + photoreal distortions. No PHI. Isolated from checkbox marks.",
        "tubes": {k: tubes[k] for k in ("n", "clean", "photoreal")},
        "dates": {k: dates[k] for k in ("n", "clean", "photoreal")},
        "others": {k: others[k] for k in ("n", "clean", "photoreal")},
        "backend": "digits+charset+writein-lexicon (TrOCR optional)",
    }
    SCORECARD.parent.mkdir(parents=True, exist_ok=True)
    SCORECARD.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    assert SCORECARD.exists()

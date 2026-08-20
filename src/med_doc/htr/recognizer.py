"""Handwriting recognizers: digits, dates, optional TrOCR, lexicon fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

DATE_RE = re.compile(
    r"\b(\d{1,2})[./\- ](\d{1,2})[./\- ](\d{2,4})\b"
)
TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")


@dataclass
class VerbalHypothesis:
    text: str
    confidence: float
    source: str


def _as_gray(crop: np.ndarray) -> np.ndarray:
    arr = np.asarray(crop)
    if arr.ndim == 3:
        return arr.mean(axis=2).astype(np.float32)
    return arr.astype(np.float32)


def _ink_mask(gray: np.ndarray, threshold: float = 140.0) -> np.ndarray:
    paper = float(np.percentile(gray, 90)) if gray.size else 255.0
    cut = min(threshold, paper * 0.55)
    mask = gray < cut
    # Drop printed header bars near the top only (not a mid-field handwritten line).
    if mask.ndim == 2 and mask.shape[0] >= 8:
        header_lim = max(2, int(mask.shape[0] * 0.20))
        row_frac = mask.mean(axis=1)
        idx = np.arange(mask.shape[0])
        mask[(idx < header_lim) & (row_frac > 0.40)] = False
    return mask


def has_ink(crop: np.ndarray | None, threshold: float = 140.0, min_frac: float | None = None) -> bool:
    """True when the crop contains a meaningful amount of dark ink.

    `min_frac` is area-adaptive: a few dozen ink pixels on a large `others` ROI
    still count. Pass an explicit fraction to keep the old small-crop behaviour.
    """
    if crop is None:
        return False
    gray = _as_gray(crop)
    if gray.size == 0 or min(gray.shape[:2]) < 4:
        return False
    h, w = gray.shape[:2]
    interior = gray[int(h * 0.08) : int(h * 0.92), int(w * 0.06) : int(w * 0.94)]
    if interior.size == 0:
        interior = gray
    mask = _ink_mask(interior, threshold)
    # Ignore printed checkbox frames: ink only on the outer ring is not handwriting.
    ih, iw = interior.shape[:2]
    inner = mask.copy()
    inner[: int(ih * 0.12), :] = False
    inner[int(ih * 0.88) :, :] = False
    inner[:, : int(iw * 0.10)] = False
    inner[:, int(iw * 0.90) :] = False
    n_ink = int(inner.sum())
    frac = float(n_ink) / float(max(inner.size, 1))
    if min_frac is not None:
        return frac >= min_frac
    adaptive = max(18.0 / float(interior.size), 0.005)
    return n_ink >= 18 and frac >= adaptive


def crop_to_ink(crop: np.ndarray, pad: int = 10) -> np.ndarray:
    """Tight RGB crop around dark ink, skipping printed header bars."""
    arr = np.asarray(crop)
    gray = _as_gray(arr)
    if gray.size == 0:
        return arr
    mask = _ink_mask(gray)
    ys, xs = np.where(mask)
    if len(ys) < 20:
        return arr
    h, w = gray.shape[:2]
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    if y1 - y0 < 4 or x1 - x0 < 4:
        return arr
    return arr[y0:y1, x0:x1]


def extract_digits(text: str) -> str:
    """Keep only digit characters from a raw string."""
    return "".join(ch for ch in (text or "") if ch.isdigit())


def extract_digits_from_crop(crop: np.ndarray | None) -> VerbalHypothesis:
    """Estimate a handwritten digit count from a tube-count crop.

    Without TrOCR this is conservative: presence of ink yields an unknown digit
    (empty string, low confidence) so fusion can fall back to Block 2 priors.
    """
    if crop is None or not has_ink(crop):
        return VerbalHypothesis(text="", confidence=0.85, source="empty")
    # Optional TrOCR path for actual digit reading
    trocr = _try_trocr(crop, max_length=8)
    if trocr is not None:
        digits = extract_digits(trocr.text)
        if digits:
            return VerbalHypothesis(text=digits, confidence=trocr.confidence, source="trocr")
        return VerbalHypothesis(text="", confidence=0.4, source="trocr-nodigit")
    return VerbalHypothesis(text="", confidence=0.45, source="ink-present")


def parse_datetime(text: str) -> tuple[str, float]:
    """Normalize a date/time string if a recognizable pattern is present."""
    raw = (text or "").strip()
    if not raw:
        return "", 0.0
    dm = DATE_RE.search(raw)
    tm = TIME_RE.search(raw)
    parts: list[str] = []
    conf = 0.0
    if dm:
        d, m, y = dm.group(1), dm.group(2), dm.group(3)
        if len(y) == 2:
            y = "20" + y
        parts.append(f"{int(d):02d}/{int(m):02d}/{y}")
        conf = 0.85
    if tm:
        parts.append(f"{int(tm.group(1)):02d}:{tm.group(2)}")
        conf = max(conf, 0.8)
    if parts:
        return " ".join(parts), conf
    return raw, 0.4


_TROCR = None
_TROCR_PROC = None
_TROCR_FAILED = False


def _try_trocr(crop: np.ndarray, max_length: int = 64) -> VerbalHypothesis | None:
    """Run microsoft/trocr-base-handwritten when transformers is installed."""
    global _TROCR, _TROCR_PROC, _TROCR_FAILED
    if _TROCR_FAILED:
        return None
    try:
        import torch
        from PIL import Image
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except Exception:
        _TROCR_FAILED = True
        return None

    if _TROCR is None:
        try:
            _TROCR_PROC = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            _TROCR = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            _TROCR.eval()
            if torch.cuda.is_available():
                _TROCR.to("cuda")
        except Exception:
            _TROCR_FAILED = True
            return None

    arr = np.asarray(crop_to_ink(crop))
    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=-1)
    else:
        rgb = arr
    image = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    if min(image.size) < 4:
        return VerbalHypothesis(text="", confidence=0.0, source="empty")

    pixel_values = _TROCR_PROC(images=image, return_tensors="pt").pixel_values
    if torch.cuda.is_available():
        pixel_values = pixel_values.to("cuda")
    with torch.no_grad():
        generated = _TROCR.generate(pixel_values, max_length=max_length)
    text = _TROCR_PROC.batch_decode(generated, skip_special_tokens=True)[0].strip()
    conf = 0.55 if text else 0.0
    return VerbalHypothesis(text=text, confidence=conf, source="trocr")


def recognize_handwriting(
    crop: np.ndarray | None,
    field_id: str,
    *,
    backend: str = "auto",
) -> VerbalHypothesis:
    """Recognize text in a handwriting crop.

    Default backend is lightweight (ink detection + empty). TrOCR is used when
    `backend` is `trocr`/`auto` and transformers is available.
    """
    if crop is None or not has_ink(crop):
        return VerbalHypothesis(text="", confidence=0.85, source="empty")

    if field_id.startswith("tube_"):
        return extract_digits_from_crop(crop)

    if backend in ("auto", "trocr"):
        trocr = _try_trocr(crop)
        if trocr is not None:
            return trocr
        if backend == "trocr":
            return VerbalHypothesis(text="", confidence=0.0, source="unavailable")

    return VerbalHypothesis(text="", confidence=0.4, source="ink-present")

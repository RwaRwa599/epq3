"""Handwriting recognizers: digits, dates, optional TrOCR, charset fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from med_doc.htr.glyphs import DATE_CHARS, TEXT_CHARS, WRITEIN_LEXICON, match_writein, read_charset_line, read_digits
from med_doc.htr.preprocess import crop_to_ink as _crop_to_ink
from med_doc.htr.preprocess import has_ink as _has_ink

DATE_RE = re.compile(
    r"\b(\d{1,2})[./\- ](\d{1,2})[./\- ](\d{2,4})\b"
)
TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")


@dataclass
class VerbalHypothesis:
    text: str
    confidence: float
    source: str
    alternatives: list[tuple[str, float]] = field(default_factory=list)


def has_ink(
    crop: np.ndarray | None,
    threshold: float = 140.0,
    min_frac: float | None = None,
    blank: np.ndarray | None = None,
) -> bool:
    return _has_ink(crop, threshold=threshold, min_frac=min_frac, blank=blank)


def crop_to_ink(crop: np.ndarray, pad: int = 10, blank: np.ndarray | None = None) -> np.ndarray:
    return _crop_to_ink(crop, pad=pad, blank=blank)


def extract_digits(text: str) -> str:
    """Keep only digit characters from a raw string."""
    return "".join(ch for ch in (text or "") if ch.isdigit())


def extract_digits_from_crop(
    crop: np.ndarray | None,
    blank: np.ndarray | None = None,
) -> VerbalHypothesis:
    """Read a tube-count digit from a small crop (prototype matcher, optional TrOCR)."""
    if crop is None or not has_ink(crop, blank=blank):
        return VerbalHypothesis(text="", confidence=0.85, source="empty")
    text, conf, source = read_digits(crop, blank=blank)
    if text:
        return VerbalHypothesis(text=text, confidence=conf, source=source)
    trocr = _try_trocr(crop, max_length=8, blank=blank)
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


def _read_date(crop: np.ndarray, blank: np.ndarray | None) -> VerbalHypothesis:
    text, conf, hyps = read_charset_line(crop, DATE_CHARS, blank=blank)
    parsed, pconf = parse_datetime(text.replace(" ", ""))
    if DATE_RE.search(parsed) or DATE_RE.search(text):
        value = parsed if DATE_RE.search(parsed) else text
        alts = [(t, s) for t, s in hyps if t != value]
        return VerbalHypothesis(
            text=value,
            confidence=max(conf, pconf),
            source="date",
            alternatives=alts,
        )
    trocr = _try_trocr(crop, max_length=24, blank=blank)
    if trocr is not None and trocr.text:
        parsed_t, p2 = parse_datetime(trocr.text)
        if DATE_RE.search(parsed_t):
            return VerbalHypothesis(text=parsed_t, confidence=max(p2, trocr.confidence), source="trocr")
        return VerbalHypothesis(text=trocr.text, confidence=trocr.confidence, source="trocr")
    if text:
        return VerbalHypothesis(text=text, confidence=conf, source="date-raw", alternatives=hyps[1:])
    return VerbalHypothesis(text="", confidence=0.45, source="ink-present")


def _read_text_box(crop: np.ndarray, blank: np.ndarray | None, backend: str) -> VerbalHypothesis:
    lex, lex_conf = match_writein(crop, blank=blank)
    text, conf, hyps = read_charset_line(crop, TEXT_CHARS, blank=blank)
    charset_known = text.replace(" ", "").upper() in WRITEIN_LEXICON
    if lex and not charset_known and lex_conf >= 0.42:
        text, conf, source = lex, max(lex_conf, 0.62), "lexicon"
    elif lex and lex_conf >= 0.55 and (not text or lex_conf >= conf):
        text, conf, source = lex, lex_conf, "lexicon"
    elif text and conf >= 0.48:
        source = "charset"
    else:
        source = ""
    trocr = None
    if backend in ("auto", "trocr"):
        trocr = _try_trocr(crop, max_length=64, blank=blank)
    alts: list[tuple[str, float]] = list(hyps)
    if lex:
        alts.append((lex, lex_conf))
    if trocr is not None and trocr.text:
        alts.insert(0, (trocr.text, trocr.confidence))
        if len(trocr.text) >= max(len(text), 3) and trocr.confidence >= conf:
            return VerbalHypothesis(
                text=trocr.text,
                confidence=trocr.confidence,
                source="trocr",
                alternatives=[h for h in alts if h[0] != trocr.text],
            )
    if source:
        return VerbalHypothesis(
            text=text,
            confidence=conf,
            source=source,
            alternatives=[h for h in alts if h[0] != text],
        )
    if backend == "trocr" and trocr is None and not text:
        return VerbalHypothesis(text="", confidence=0.0, source="unavailable")
    if text:
        return VerbalHypothesis(text=text, confidence=conf, source="charset", alternatives=alts[1:])
    return VerbalHypothesis(text="", confidence=0.45, source="ink-present")


_TROCR = None
_TROCR_PROC = None
_TROCR_FAILED = False


def _try_trocr(
    crop: np.ndarray,
    max_length: int = 64,
    blank: np.ndarray | None = None,
) -> VerbalHypothesis | None:
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

    arr = np.asarray(crop_to_ink(crop, blank=blank))
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
    # Sequence length is a weak proxy; still better than a constant 0.55 on empty.
    conf = float(np.clip(0.35 + 0.08 * min(len(text), 8), 0.0, 0.9)) if text else 0.0
    return VerbalHypothesis(text=text, confidence=conf, source="trocr")


def recognize_handwriting(
    crop: np.ndarray | None,
    field_id: str,
    *,
    backend: str = "auto",
    blank: np.ndarray | None = None,
) -> VerbalHypothesis:
    """Recognize text in a handwriting crop.

    Tubes → digit prototypes. Dates → charset + grammar. Others → charset
    lines (n-best) with optional TrOCR. KG matching is Block 4.
    """
    if crop is None or not has_ink(crop, blank=blank):
        return VerbalHypothesis(text="", confidence=0.85, source="empty")

    if field_id.startswith("tube_"):
        return extract_digits_from_crop(crop, blank=blank)

    if field_id in {"received_at", "date", "sample_received"}:
        return _read_date(crop, blank)

    return _read_text_box(crop, blank, backend)

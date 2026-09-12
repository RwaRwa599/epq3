"""Prototype glyph reader for tube digits and constrained charset lines.

No torch required. Prototypes are rendered with OpenCV Hershey fonts and
matched by cosine similarity on a 24×24 ink patch plus axis projections.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from med_doc.htr.preprocess import crop_to_ink, ink_mask, split_char_boxes, split_line_bands, working_gray

DIGITS = "0123456789"
DATE_CHARS = "0123456789/:.-"
TEXT_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-"
WRITEIN_LEXICON = (
    "AFP",
    "CEA",
    "CBC",
    "HBA1C",
    "LIPID",
    "URIC",
    "ALT",
    "AST",
    "TSH",
    "PSA",
    "CA125",
    "GLUCOSE",
)

SIZE = 24


def _to_u8(gray: np.ndarray) -> np.ndarray:
    arr = np.asarray(gray)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _ink_patch(gray: np.ndarray) -> np.ndarray:
    work = _to_u8(gray)
    mask = ink_mask(work)
    ys, xs = np.where(mask)
    if len(ys) < 4:
        return np.zeros((SIZE, SIZE), dtype=np.float32)
    y0, y1 = max(0, int(ys.min()) - 1), min(work.shape[0], int(ys.max()) + 2)
    x0, x1 = max(0, int(xs.min()) - 1), min(work.shape[1], int(xs.max()) + 2)
    crop = work[y0:y1, x0:x1]
    m = ink_mask(crop)
    patch = np.where(m, 0.0, 255.0).astype(np.float32)
    return cv2.resize(patch, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def glyph_features(gray: np.ndarray) -> np.ndarray:
    patch = _ink_patch(gray)
    ink = (patch < 140).astype(np.float32)
    pix = cv2.resize(ink, (12, 12), interpolation=cv2.INTER_AREA).ravel()
    hx = ink.mean(axis=0)
    hy = ink.mean(axis=1)
    vec = np.concatenate([pix, hx, hy])
    n = float(np.linalg.norm(vec)) + 1e-6
    return vec / n


def _render_char(ch: str, *, thickness: int, scale: float, dx: int, dy: int) -> np.ndarray:
    canvas = np.full((40, 32), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX if ch.isalnum() else cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _base = cv2.getTextSize(ch, font, scale, thickness)
    x = max(1, (32 - tw) // 2 + dx)
    y = max(th + 2, (40 + th) // 2 + dy)
    cv2.putText(canvas, ch, (x, y), font, scale, 20, thickness, cv2.LINE_AA)
    return canvas


@lru_cache(maxsize=4)
def _prototypes(charset: str) -> dict[str, np.ndarray]:
    means: dict[str, np.ndarray] = {}
    for ch in charset:
        feats = []
        for thickness in (1, 2):
            for scale in (0.7, 0.85, 1.0):
                for dx, dy in ((0, 0), (1, 0), (0, 1), (-1, 0)):
                    img = _render_char(ch, thickness=thickness, scale=scale, dx=dx, dy=dy)
                    feats.append(glyph_features(img))
        means[ch] = np.mean(np.stack(feats), axis=0)
        n = float(np.linalg.norm(means[ch])) + 1e-6
        means[ch] = means[ch] / n
    return means


def classify_glyph(gray: np.ndarray, charset: str = DIGITS) -> tuple[str, float, str]:
    """Return (best_char, confidence, second_char). Confidence is a margin in [0, 1]."""
    feat = glyph_features(gray)
    proto = _prototypes(charset)
    best_ch, best_s = "?", -1.0
    second_ch, second_s = "?", -1.0
    for ch, mean in proto.items():
        s = float(np.dot(feat, mean))
        if s > best_s:
            second_ch, second_s = best_ch, best_s
            best_ch, best_s = ch, s
        elif s > second_s:
            second_ch, second_s = ch, s
    margin = max(0.0, best_s - second_s)
    conf = float(np.clip(0.45 + 0.9 * margin + 0.25 * max(best_s, 0.0), 0.0, 0.99))
    if best_s < 0.35:
        return "?", 0.2, second_ch
    return best_ch, conf, second_ch


def _split_two_digits(gray: np.ndarray) -> list[np.ndarray]:
    mask = ink_mask(gray)
    boxes = split_char_boxes(mask, min_width=2)
    h, w = gray.shape[:2]
    if len(boxes) >= 2:
        out = []
        for x0, x1 in boxes[:2]:
            pad = 1
            out.append(gray[:, max(0, x0 - pad) : min(w, x1 + pad)])
        return out
    if w >= int(h * 1.25):
        mid = w // 2
        return [gray[:, :mid], gray[:, mid:]]
    return [gray]


def read_digits(crop: np.ndarray, blank: np.ndarray | None = None) -> tuple[str, float, str]:
    """Read 1–2 handwritten tube-count digits. Empty string if rejected."""
    tight = crop_to_ink(crop, pad=4, blank=blank)
    gray = working_gray(tight, None)
    parts = _split_two_digits(gray)
    chars: list[str] = []
    confs: list[float] = []
    for part in parts:
        if part.size < 16 or min(part.shape[:2]) < 4:
            continue
        ch, conf, _ = classify_glyph(part, DIGITS)
        if ch in DIGITS and conf >= 0.48:
            chars.append(ch)
            confs.append(conf)
    if not chars:
        return "", 0.35, "digit-reject"
    text = "".join(chars)[:2]
    return text, float(min(confs) if confs else 0.4), "digits"


def _render_word_gray(word: str) -> np.ndarray:
    width = max(80, 22 * len(word) + 24)
    canvas = np.full((52, width), 255, dtype=np.uint8)
    cv2.putText(canvas, word, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 20, 2, cv2.LINE_AA)
    return canvas


def _letterbox_ink(mask: np.ndarray, height: int = 32, width: int = 160) -> np.ndarray:
    ink = mask.astype(np.float32)
    if ink.size == 0:
        return np.zeros((height, width), dtype=np.float32)
    h, w = ink.shape[:2]
    scale = min(height / max(h, 1), width / max(w, 1))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = cv2.resize(ink, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width), dtype=np.float32)
    y = (height - nh) // 2
    x = (width - nw) // 2
    canvas[y : y + nh, x : x + nw] = small
    return canvas


def _line_vector(gray: np.ndarray) -> np.ndarray:
    work = _to_u8(gray)
    mask = ink_mask(work)
    ys, xs = np.where(mask)
    if len(ys) >= 4:
        mask = mask[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
    vec = _letterbox_ink(mask).ravel()
    n = float(np.linalg.norm(vec)) + 1e-6
    return vec / n


def match_writein(
    crop: np.ndarray,
    lexicon: tuple[str, ...] = WRITEIN_LEXICON,
    blank: np.ndarray | None = None,
) -> tuple[str, float]:
    """Whole-line cosine match against rendered write-in tokens (not KG assume)."""
    tight = crop_to_ink(crop, pad=4, blank=blank)
    gray = working_gray(tight, None)
    feat = _line_vector(gray)
    best, best_s = "", -1.0
    second = -1.0
    for word in lexicon:
        proto = _render_word_gray(word)
        s = float(np.dot(feat, _line_vector(proto)))
        if s > best_s:
            second = best_s
            best, best_s = word, s
        elif s > second:
            second = s
    margin = max(0.0, best_s - second)
    conf = float(np.clip(0.4 + 1.1 * margin + 0.25 * max(best_s, 0.0), 0.0, 0.99))
    if best_s < 0.55 or margin < 0.02:
        return "", 0.2
    return best, conf


def read_charset_line(
    crop: np.ndarray,
    charset: str,
    blank: np.ndarray | None = None,
) -> tuple[str, float, list[tuple[str, float]]]:
    """Segment lines/glyphs and classify each against ``charset``."""
    tight = crop_to_ink(crop, pad=4, blank=blank)
    gray = working_gray(tight, None)
    mask = ink_mask(gray)
    lines = split_line_bands(mask, min_height=6)
    if not lines:
        lines = [(0, gray.shape[0])]
    texts: list[str] = []
    confs: list[float] = []
    alt_lines: list[str] = []
    for y0, y1 in lines:
        band = gray[y0:y1]
        bmask = mask[y0:y1]
        boxes = split_char_boxes(bmask, min_width=2)
        row = []
        arow = []
        prev_x1 = None
        for x0, x1 in boxes:
            if prev_x1 is not None and x0 - prev_x1 >= 8:
                row.append(" ")
                arow.append(" ")
            glyph = band[:, x0:x1]
            ch, conf, second = classify_glyph(glyph, charset)
            if ch == "?":
                continue
            row.append(ch)
            arow.append(second if second in charset else ch)
            confs.append(conf)
            prev_x1 = x1
        if row:
            texts.append("".join(row).strip())
            alt_lines.append("".join(arow).strip())
    text = "\n".join(t for t in texts if t)
    alt = "\n".join(t for t in alt_lines if t)
    if not text:
        return "", 0.3, []
    conf = float(np.median(confs)) if confs else 0.4
    hyps = [(text, conf)]
    if alt and alt != text:
        hyps.append((alt, max(0.2, conf - 0.12)))
    return text, conf, hyps

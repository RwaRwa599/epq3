#!/usr/bin/env python3
"""Crop ~10.5 cm off the top of lab-form scans; keep the rest of the page.

Then check that "Clinical Information" and a column header (e.g. HAEMATOLOGY)
are still fully inside the crop. If 10.5 cm would cut those, the cut moves up.

    python3 -m pip install pillow numpy opencv-python-headless
    python3 crop_labform_pages.py /path/to/pages --out /path/to/cropped --debug
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
CM_PER_INCH = 2.54
A4_HEIGHT_CM = 29.7
DEFAULT_TOP_CM = 10.5

CLINICAL_PHRASES = ("Clinical Information", "CLINICAL INFORMATION", "INFORMATION")
HEADER_PHRASES = (
    "HAEMATOLOGY",
    "CHECK-UP",
    "PROFILE",
    "CLINICAL CHEMISTRY",
    "IMMUNOLOGY",
    "ENDOCRINOLOGY",
    "TUMOR MARKERS",
    "URINE",
)


@dataclass
class TokenHit:
    phrase: str
    score: float
    x: int
    y: int
    w: int
    h: int

    @property
    def y0(self) -> int:
        return self.y

    @property
    def y1(self) -> int:
        return self.y + self.h


def image_dpi(im: Image.Image, override: float | None) -> float | None:
    if override and override > 1:
        return float(override)
    info = im.info.get("dpi")
    if isinstance(info, tuple) and info[0]:
        return float(info[1] or info[0])
    if isinstance(info, (int, float)) and info:
        return float(info)
    return None


def top_px_from_cm(
    height_px: int,
    top_cm: float,
    *,
    dpi: float | None,
    page_height_cm: float = A4_HEIGHT_CM,
) -> int:
    if dpi and dpi > 1:
        return int(round(top_cm / CM_PER_INCH * dpi))
    return int(round(height_px * (top_cm / page_height_cm)))


def _cv2():
    import cv2
    import numpy as np

    return cv2, np


def locate_phrase(rgb, phrase: str, *, min_score: float = 0.38) -> TokenHit | None:
    """Best NCC of a Hershey rendering of ``phrase`` on an RGB page."""
    try:
        cv2, np = _cv2()
    except ImportError:
        return None
    arr = np.asarray(rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
    gh, gw = gray.shape[:2]
    if min(gh, gw) < 16:
        return None
    canvas = np.full((56, max(80, 18 * len(phrase) + 32)), 255, dtype=np.uint8)
    cv2.putText(canvas, phrase, (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, 20, 1, cv2.LINE_AA)
    best: TokenHit | None = None
    for scale in (0.45, 0.6, 0.8, 1.0, 1.25, 1.6, 2.1):
        th, tw = max(10, int(canvas.shape[0] * scale)), max(16, int(canvas.shape[1] * scale))
        if th >= gh or tw >= gw:
            continue
        small = cv2.resize(canvas, (tw, th), interpolation=cv2.INTER_AREA)
        for work in (small, 255 - small):
            ncc = cv2.matchTemplate(
                gray.astype(np.float32), work.astype(np.float32), cv2.TM_CCOEFF_NORMED
            )
            if ncc.size == 0:
                continue
            score = float(ncc.max())
            if score < min_score:
                continue
            _, _, _, maxloc = cv2.minMaxLoc(ncc)
            hit = TokenHit(phrase=phrase, score=score, x=int(maxloc[0]), y=int(maxloc[1]), w=tw, h=th)
            if best is None or hit.score > best.score:
                best = hit
    return best


def locate_required(rgb) -> dict[str, TokenHit | None]:
    clinical = None
    for phrase in CLINICAL_PHRASES:
        hit = locate_phrase(rgb, phrase)
        if hit is not None and (clinical is None or hit.score > clinical.score):
            clinical = hit
    header = None
    for phrase in HEADER_PHRASES:
        hit = locate_phrase(rgb, phrase)
        if hit is not None and (header is None or hit.score > header.score):
            header = hit
    return {"clinical": clinical, "header": header}


def safe_top_px(hits: dict[str, TokenHit | None], *, pad: int = 8) -> int | None:
    """Highest cut that still keeps clinical-info + a column header intact."""
    keep: list[int] = []
    for key in ("clinical", "header"):
        hit = hits.get(key)
        if hit is not None:
            keep.append(max(0, hit.y0 - pad))
    if not keep:
        return None
    return min(keep)


def evaluate_crop(rgb_crop, *, min_score: float = 0.38) -> dict:
    hits = locate_required(rgb_crop)
    clinical = hits["clinical"]
    header = hits["header"]
    clinical_ok = clinical is not None and clinical.score >= min_score and clinical.y0 >= 0
    header_ok = header is not None and header.score >= min_score and header.y0 >= 0
    return {
        "clinical_information": clinical_ok,
        "column_headers": header_ok,
        "ok": bool(clinical_ok and header_ok),
        "clinical": None
        if clinical is None
        else {"phrase": clinical.phrase, "score": round(clinical.score, 3), "y": clinical.y0},
        "header": None
        if header is None
        else {"phrase": header.phrase, "score": round(header.score, 3), "y": header.y0},
    }


def plan_crop(
    im: Image.Image,
    *,
    top_cm: float = DEFAULT_TOP_CM,
    dpi: float | None = None,
    page_height_cm: float = A4_HEIGHT_CM,
    pad: int = 8,
) -> dict:
    w, h = im.size
    proposed = min(h - 1, max(0, top_px_from_cm(h, top_cm, dpi=image_dpi(im, dpi), page_height_cm=page_height_cm)))
    hits = locate_required(im)
    safe = safe_top_px(hits, pad=pad)
    y0 = proposed
    adjusted = False
    reason = "top_cm"
    if safe is not None and proposed > safe:
        y0 = safe
        adjusted = True
        reason = "pulled_up_to_keep_labels"
    crop = im.crop((0, y0, w, h))
    eval_after = evaluate_crop(crop)
    return {
        "proposed_top_px": proposed,
        "top_px": y0,
        "adjusted": adjusted,
        "reason": reason,
        "safe_top_px": safe,
        "page": [w, h],
        "eval": eval_after,
        "hits_before": {
            k: None
            if v is None
            else {"phrase": v.phrase, "score": round(v.score, 3), "y": v.y0}
            for k, v in hits.items()
        },
    }


def _draw_cut_guide(im: Image.Image, y0: int) -> Image.Image:
    """Full page with the discarded top washed red so it cannot look like the crop."""
    vis = im.copy().convert("RGBA")
    w, h = vis.size
    veil = Image.new("RGBA", (w, max(1, y0)), (220, 40, 40, 140))
    vis.paste(veil, (0, 0), veil)
    draw = ImageDraw.Draw(vis)
    draw.line((0, y0, w, y0), fill=(255, 128, 0, 255), width=8)
    draw.rectangle((0, y0, w - 1, h - 1), outline=(255, 128, 0, 255), width=6)
    caption = f"REMOVED {y0}px  |  KEEP below  |  crop is the other PNG, not this guide"
    draw.rectangle((8, 8, min(w - 8, 8 + 7 * len(caption)), 36), fill=(0, 0, 0, 180))
    draw.text((14, 14), caption, fill=(255, 255, 255, 255))
    return vis.convert("RGB")


def crop_one(
    path: Path,
    out_dir: Path,
    *,
    top_cm: float,
    dpi: float | None,
    page_height_cm: float,
    debug: bool,
) -> dict:
    im = Image.open(path).convert("RGB")
    plan = plan_crop(im, top_cm=top_cm, dpi=dpi, page_height_cm=page_height_cm)
    w, h = im.size
    y0 = int(plan["top_px"])
    cropped = im.crop((0, y0, w, h))
    if cropped.size[1] == h:
        raise RuntimeError(f"crop did not change height ({w}x{h}); top_px={y0}")
    dest = out_dir / f"{path.stem}.png"
    cropped.save(dest)
    if debug:
        (out_dir / "debug").mkdir(parents=True, exist_ok=True)
        _draw_cut_guide(im, y0).save(out_dir / "debug" / f"{path.stem}_GUIDE_fullpage.png")
    row = {
        "doc_id": path.stem,
        "path": dest.name,
        "input_size": [w, h],
        "output_size": [cropped.size[0], cropped.size[1]],
        **plan,
    }
    status = "ok" if plan["eval"]["ok"] else "eval_failed"
    used_dpi = image_dpi(im, dpi)
    print(
        f"{status} {dest}  {w}x{h} → {cropped.size[0]}x{cropped.size[1]}  "
        f"removed top {y0}px ({top_cm}cm"
        f"{f' @{used_dpi:.0f}dpi' if used_dpi else ' as A4 fraction'}"
        f"{', pulled up' if plan['adjusted'] else ''})  "
        f"clinical={plan['eval']['clinical_information']} "
        f"headers={plan['eval']['column_headers']}"
    )
    print(f"  open the cropped file, not debug/: {dest}")
    return row


def collect(src: Path) -> list[Path]:
    if src.is_file() and src.suffix.lower() in SUFFIXES:
        return [src]
    if src.is_dir():
        return sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    raise SystemExit(f"Not a folder or image: {src}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path, help="Folder of PNG/JPG pages, or one image")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top-cm", type=float, default=DEFAULT_TOP_CM, dest="top_cm")
    p.add_argument("--dpi", type=float, default=None, help="Raster DPI; default = PNG dpi or A4 fraction")
    p.add_argument("--page-height-cm", type=float, default=A4_HEIGHT_CM, dest="page_height_cm")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    files = collect(args.inputs)
    if not files:
        raise SystemExit(f"No images in {args.inputs}")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [
        crop_one(
            path,
            args.out,
            top_cm=args.top_cm,
            dpi=args.dpi,
            page_height_cm=args.page_height_cm,
            debug=args.debug,
        )
        for path in files
    ]
    manifest = {
        "top_cm": args.top_cm,
        "total": len(rows),
        "ok": sum(1 for r in rows if r["eval"]["ok"]),
        "documents": rows,
    }
    (args.out / "crop_eval.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done: {manifest['ok']}/{manifest['total']} passed label check → {args.out}")
    failed = [r["doc_id"] for r in rows if not r["eval"]["ok"]]
    if failed:
        print("eval_failed:", ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

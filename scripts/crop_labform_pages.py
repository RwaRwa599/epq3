#!/usr/bin/env python3
"""Crop the name-header strip off lab-form scans; keep the rest of the page.

No med_doc / LayoutParser install. One file. Needs Pillow:

    python3 -m pip install pillow
    python3 crop_labform_pages.py /path/to/pages --out /path/to/cropped --debug

Default band (v1): y 0.08–1.0. Drops page top through the start of Clinical
Information (~1cm above column headers). Keeps tubes / office at the bottom.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def crop_one(
    path: Path,
    out_dir: Path,
    *,
    top: float,
    bottom: float,
    left: float,
    right: float,
    debug: bool,
) -> None:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    x0, y0 = int(left * w), int(top * h)
    x1, y1 = int(right * w), int(bottom * h)
    x1, y1 = max(x0 + 1, x1), max(y0 + 1, y1)
    cropped = im.crop((x0, y0, x1, y1))
    dest = out_dir / f"{path.stem}.png"
    cropped.save(dest)
    if debug:
        vis = im.copy()
        draw = ImageDraw.Draw(vis)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 128, 0), width=6)
        (out_dir / "debug").mkdir(parents=True, exist_ok=True)
        vis.save(out_dir / "debug" / f"{path.stem}_boxes.png")
    print(f"wrote {dest}  ({w}x{h} → {cropped.size[0]}x{cropped.size[1]})")


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
    p.add_argument("--top", type=float, default=0.08)
    p.add_argument("--bottom", type=float, default=1.0)
    p.add_argument("--left", type=float, default=0.0)
    p.add_argument("--right", type=float, default=1.0)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    files = collect(args.inputs)
    if not files:
        raise SystemExit(f"No images in {args.inputs}")
    args.out.mkdir(parents=True, exist_ok=True)
    for path in files:
        crop_one(
            path,
            args.out,
            top=args.top,
            bottom=args.bottom,
            left=args.left,
            right=args.right,
            debug=args.debug,
        )
    print(f"done: {len(files)} file(s) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

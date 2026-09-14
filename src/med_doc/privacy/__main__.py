"""Batch-crop lab photos to a configured layout band.

Examples::

    python -m med_doc.privacy photos/ --out cropped/
    python -m med_doc.privacy photos/ --out cropped/ \\
        --config configs/layout_crop.layoutparser.json --debug
    python -m med_doc.privacy photos/ --out cropped/ --backend layoutparser_then_template \\
        --keep Table List --drop Title Figure Text --combine largest --debug
"""

from __future__ import annotations

import argparse
import json

from med_doc.privacy.layout_crop import CropConfig, crop_batch, load_crop_config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m med_doc.privacy")
    p.add_argument("inputs", help="Folder, ZIP, or one image")
    p.add_argument("--out", required=True, help="Output folder of cropped PNGs")
    p.add_argument("--config", default=None, help="JSON (default configs/layout_crop.json)")
    p.add_argument(
        "--backend",
        choices=["template", "layoutparser", "layoutparser_then_template"],
        default=None,
    )
    p.add_argument("--keep", nargs="*", default=None, help="LayoutParser types to keep")
    p.add_argument("--drop", nargs="*", default=None, help="LayoutParser types to drop")
    p.add_argument("--combine", choices=["union", "largest", "vertical_span"], default=None)
    p.add_argument("--top", type=float, default=None, help="Template band top (0–1)")
    p.add_argument("--bottom", type=float, default=None, help="Template band bottom (0–1)")
    p.add_argument("--left", type=float, default=None)
    p.add_argument("--right", type=float, default=None)
    p.add_argument("--score", type=float, default=None, help="LayoutParser min score")
    p.add_argument("--min-area", type=float, default=None, dest="min_area", help="Min box area as fraction of page")
    p.add_argument("--debug", action="store_true", help="Write debug/<id>_boxes.png (keep=green, drop=red)")
    args = p.parse_args(argv)

    cfg = load_crop_config(args.config)
    updates: dict = {}
    if args.backend:
        updates["backend"] = args.backend
    if args.keep is not None:
        updates["keep_types"] = args.keep
    if args.drop is not None:
        updates["drop_types"] = args.drop
    if args.combine:
        updates["combine"] = args.combine
    if args.score is not None:
        updates["score_threshold"] = args.score
    if args.min_area is not None:
        updates["min_area_frac"] = args.min_area
    tmpl = dict(cfg.template.model_dump())
    for key in ("top", "bottom", "left", "right"):
        val = getattr(args, key)
        if val is not None:
            tmpl[key] = val
    updates["template"] = tmpl
    cfg = CropConfig.model_validate({**cfg.model_dump(), **updates})

    out = crop_batch(args.inputs, args.out, config=cfg, debug=args.debug)
    print(json.dumps(out["manifest"], indent=2))
    return 0 if out["manifest"]["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

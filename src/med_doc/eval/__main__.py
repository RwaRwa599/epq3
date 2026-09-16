"""CLI: export labeled crops, refit mark logreg, crop QA, model inventory.

Examples::

    python -m med_doc.eval export --block1 block1.zip --gold data/labels/examples/IMG_7596.json --out data/labels/crops
    python -m med_doc.eval train --crops data/labels/crops --write data/labels/mark_weights.json
    python -m med_doc.eval crop-qa --block1 block1.zip --gold data/labels/examples/IMG_7596.json --field ca125
    python -m med_doc.eval vision-bakeoff --block1 out/block1.zip --gold data/labels/examples/IMG_7596.json --models qwen2.5vl:7b,llama3.2-vision
    python -m med_doc.eval text-bakeoff --hypotheses out/b3 --gold data/labels/examples/IMG_7596.json --models qwen2.5:7b-instruct
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

from med_doc.eval.bakeoff import text_bakeoff, vision_bakeoff
from med_doc.eval.calibrate_marks import train_and_save
from med_doc.eval.crop_qa import crop_qa
from med_doc.eval.export_crops import export_labeled_crops
from med_doc.eval.models import models_in_use


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m med_doc.eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="Block 1 ZIP + gold field ids → empty/tick PNGs")
    e.add_argument("--block1", required=True)
    e.add_argument("--gold", required=True)
    e.add_argument("--out", default="data/labels/crops")

    t = sub.add_parser("train", help="Refit LOGREG_W/B (synthetic + optional clinic crops)")
    t.add_argument("--crops", default=None, help="data/labels/crops (gitignored clinic PNGs)")
    t.add_argument("--write", default="data/labels/mark_weights.json")
    t.add_argument("--empty-repeats", type=int, default=8)

    q = sub.add_parser("crop-qa", help="Is a gold tick a shifted label crop?")
    q.add_argument("--block1", required=True)
    q.add_argument("--gold", required=True)
    q.add_argument("--field", default=None)

    sub.add_parser("models", help="Print which models actually run")

    vb = sub.add_parser("vision-bakeoff", help="Local VL models on page+HW (+ disagreement crops) vs gold/3a")
    vb.add_argument("--block1", required=True)
    vb.add_argument("--gold", required=True, help="Gitignored clinic gold or data/labels/examples")
    vb.add_argument("--models", default="qwen2.5vl:7b")
    vb.add_argument("--out", default=None)

    tb = sub.add_parser("text-bakeoff", help="Instruct models on frozen hypotheses n-best (not ticks)")
    tb.add_argument("--hypotheses", required=True)
    tb.add_argument("--gold", required=True)
    tb.add_argument("--models", default="qwen2.5:7b-instruct")
    tb.add_argument("--out", default=None)

    args = p.parse_args(argv)
    if args.cmd == "export":
        stats = export_labeled_crops(args.block1, args.gold, args.out)
        print(json.dumps(stats, indent=2))
        return 0
    if args.cmd == "train":
        stats = train_and_save(args.write, crop_dir=args.crops, empty_repeats=args.empty_repeats)
        print(json.dumps(stats, indent=2))
        print("Set MED_DOC_MARK_WEIGHTS or keep data/labels/mark_weights.json (gitignored).")
        return 0
    if args.cmd == "crop-qa":
        rows = crop_qa(args.block1, args.gold, field_id=args.field)
        print(json.dumps(rows, indent=2))
        n_shift = sum(1 for r in rows if r["cause"] == "block1_shift")
        if n_shift:
            print(f"{n_shift} crop(s) look like Block 1 shift — fix alignment/1c before HTR.")
        return 0
    if args.cmd == "models":
        print(json.dumps(models_in_use(), indent=2))
        return 0
    if args.cmd == "vision-bakeoff":
        models = [m.strip() for m in str(args.models).split(",") if m.strip()]
        result = vision_bakeoff(args.block1, args.gold, models)
        print(json.dumps(result, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0
    if args.cmd == "text-bakeoff":
        models = [m.strip() for m in str(args.models).split(",") if m.strip()]
        result = text_bakeoff(args.hypotheses, args.gold, models)
        print(json.dumps(result, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0
    p.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

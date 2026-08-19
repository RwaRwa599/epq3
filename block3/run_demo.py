#!/usr/bin/env python3
"""CLI for Block 3: ingest Block 1 ZIP, run verbal/nonverbal, emit hypotheses ZIP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from med_doc.htr import process_from_block1
from med_doc.htr.nonverbal import classify_mark_nonverbal, paddle_available
from med_doc.htr.verbal import recognize_verbal, trocr_available
from med_doc.kg import KnowledgeGraph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block 3 CLI — nonverbal marks (Paddle) + verbal HTR (TrOCR)"
    )
    parser.add_argument("--input-zip", type=str, default=None, help="Block 1 ZIP")
    parser.add_argument(
        "--output-zip",
        type=str,
        default="block3_predictions_batch.zip",
    )
    parser.add_argument(
        "--mode",
        choices=["nonverbal", "verbal", "both"],
        default="both",
    )
    parser.add_argument("--backend", choices=["auto", "lexicon", "trocr"], default="auto")
    parser.add_argument("--kg-path", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  BLOCK 3: NONVERBAL MARKS + VERBAL HANDWRITING")
    print("=" * 70)
    print(f"  PaddleOCR: {paddle_available()}    TrOCR: {trocr_available()}")

    kg = KnowledgeGraph.load(args.kg_path)
    print(f"\n[+] Loaded Block 2 KG: version='{kg.version}' ({len(kg.catalogue)} tests)")

    if args.input_zip:
        res = process_from_block1(
            args.input_zip,
            output_zip=args.output_zip,
            kg=kg,
            backend=args.backend,
            mode=args.mode,
        )
        manifest = res["manifest"]
        print("\n" + "=" * 70)
        print("✓ Batch complete")
        print(f"  Mode:            {args.mode}")
        print(f"  Total documents: {manifest['total_documents']}")
        print(f"  Need HiTL:       {manifest['hitl_documents']}")
        print(f"  Output ZIP:      {res['output_zip']}")
        print("=" * 70)
        return

    empty = np.full((32, 32, 3), 245, dtype=np.uint8)
    empty[0:2, :] = 20
    empty[-2:, :] = 20
    empty[:, 0:2] = 20
    empty[:, -2:] = 20
    tick = empty.copy()
    for i in range(5, 27):
        tick[i, i] = 15
        tick[i, min(31, i + 1)] = 15
    print("\n[*] Nonverbal smoke test")
    print(f"    empty -> {classify_mark_nonverbal(empty, 'cbc').model_dump()}")
    print(f"    tick  -> {classify_mark_nonverbal(tick, 'cbc').model_dump()}")
    print("\n[*] Verbal smoke test (empty others crop)")
    print(f"    {recognize_verbal(empty, 'others').model_dump()}")
    print("\nPass --input-zip block1_normalized_batch.zip to run a real batch.")
    print("=" * 70)


if __name__ == "__main__":
    main()

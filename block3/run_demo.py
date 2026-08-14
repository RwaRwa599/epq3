#!/usr/bin/env python3
"""CLI for Block 3: ingest Block 2 ZIP, classify marks, fuse handwriting, export Block 4 ZIP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from med_doc.htr import process_batch_from_block2
from med_doc.kg import KnowledgeGraph
from med_doc.htr.marks import classify_mark
from med_doc.htr.fusion import fuse_handwriting
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block 3 CLI — HTR, mark classification, and prior fusion"
    )
    parser.add_argument(
        "--input-zip",
        type=str,
        default=None,
        help="Path to Block 2 ZIP (block2_validated_batch.zip)",
    )
    parser.add_argument(
        "--output-zip",
        type=str,
        default="block3_predictions_batch.zip",
        help="Path to save Block 3 ZIP for Block 4 / LIS",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "lexicon", "trocr"],
        default="auto",
        help="HTR backend (lexicon is CPU-only; trocr needs transformers)",
    )
    parser.add_argument(
        "--kg-path",
        type=str,
        default=None,
        help="Optional custom Knowledge Graph JSON",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  BLOCK 3: HANDWRITING RECOGNITION & PRIOR FUSION ENGINE")
    print("=" * 70)

    kg = KnowledgeGraph.load(args.kg_path)
    print(f"\n[+] Loaded Knowledge Graph: version='{kg.version}' ({len(kg.catalogue)} tests)")

    if args.input_zip:
        res = process_batch_from_block2(
            args.input_zip,
            output_zip=args.output_zip,
            kg=kg,
            backend=args.backend,
        )
        manifest = res["manifest"]
        print("\n" + "=" * 70)
        print("✓ Batch HTR complete")
        print(f"  Total documents: {manifest['total_documents']}")
        print(f"  Valid:           {manifest['valid_documents']}")
        print(f"  Need HiTL:       {manifest['hitl_documents']}")
        print(f"  Output ZIP:      {res['output_zip']}")
        print("=" * 70)
        return

    # Offline smoke demo without a ZIP
    empty = np.full((32, 32, 3), 245, dtype=np.uint8)
    empty[0:2, :] = 20
    empty[-2:, :] = 20
    empty[:, 0:2] = 20
    empty[:, -2:] = 20
    tick = empty.copy()
    for i in range(5, 27):
        tick[i, i] = 15
        tick[i, min(31, i + 1)] = 15
        tick[min(31, i + 1), i] = 15
    print("\n[*] Mark classifier smoke test")
    print(f"    empty -> {classify_mark(empty, 'cbc').model_dump()}")
    print(f"    tick  -> {classify_mark(tick, 'cbc').model_dump()}")

    fused = fuse_handwriting(
        "others",
        "cultur and sensitivty",
        0.5,
        "demo",
        kg=kg,
        ticked_ids=["cbc"],
    )
    print("\n[*] Prior fusion smoke test (OTHERS = 'cultur and sensitivty')")
    print(f"    canonical={fused.canonical_value} id={fused.canonical_id} conf={fused.confidence}")
    print("\nPass --input-zip block2_validated_batch.zip to run a real batch.")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Interactive CLI demo for Block 2: Clinical Knowledge Graph, Prior Engine & Batch ZIP Ingestion."""

import argparse
import json
import sys
from pathlib import Path

# Add local directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent))

from med_doc.kg import KnowledgeGraph, process_batch_from_block1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block 2 CLI Demo — Clinical Knowledge Graph, Prior Engine & Batch Ingestion"
    )
    parser.add_argument(
        "--input-zip",
        type=str,
        default=None,
        help="Path to Block 1 normalized batch ZIP file (e.g. block1_normalized_batch.zip)",
    )
    parser.add_argument(
        "--output-zip",
        type=str,
        default="block2_validated_batch.zip",
        help="Path to save validated Block 3 ZIP file",
    )
    parser.add_argument(
        "--kg-path",
        type=str,
        default=None,
        help="Path to custom Knowledge Graph JSON file (defaults to packaged v1 KG)",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["cbc", "profile_lipid", "glucose_fasting"],
        help="List of ticked test or profile IDs for single-sheet simulation",
    )
    parser.add_argument(
        "--tubes",
        nargs="+",
        default=["EDTA:1", "CB:1", "Fl:1"],
        help="Observed tube counts in format TubeName:Count (e.g. EDTA:1 CB:1 Fl:1)",
    )
    parser.add_argument(
        "--resolve",
        type=str,
        default="SGPT",
        help="Medical alias or acronym to resolve to canonical field ID",
    )
    parser.add_argument(
        "--fuzzy",
        type=str,
        default="cultur and sensitivty",
        help="Messy doctor handwriting query to fuzzy match",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  BLOCK 2: CLINICAL KNOWLEDGE GRAPH & PRIOR VALIDATION ENGINE")
    print("=" * 70)

    kg = KnowledgeGraph.load(args.kg_path)
    print(f"\n[+] Loaded Knowledge Graph: version='{kg.version}' ({len(kg.catalogue)} catalogue tests)")
    print(f"    Disclaimer: {kg.disclaimer}\n")

    # If batch ZIP input is provided
    if args.input_zip:
        print(f"[*] Processing Block 1 Batch ZIP: {args.input_zip}")
        res = process_batch_from_block1(
            args.input_zip,
            output_zip=args.output_zip,
            kg=kg,
        )
        manifest = res["manifest"]
        print("\n" + "=" * 70)
        print("✓ Batch Validation Complete!")
        print(f"  Total Processed Documents: {manifest['total_documents']}")
        print(f"  Valid Documents:           {manifest['valid_documents']}")
        print(f"  Output ZIP for Block 3:    {res['output_zip']}")
        print("=" * 70)
        return

    # Otherwise run single interactive demo
    resolved_id = kg.resolve_alias(args.resolve)
    print(f"[*] Alias Resolution: '{args.resolve}' -> '{resolved_id}'")

    fuzzy_candidates = kg.fuzzy_match_catalogue(args.fuzzy, top_k=3)
    print(f"[*] Fuzzy Match for write-in '{args.fuzzy}':")
    for rank, c in enumerate(fuzzy_candidates, 1):
        print(f"    {rank}. {c.value} (id: {c.canonical_id}) - Score: {c.score:.3f} [Tier {c.tier}]")

    print(f"\n[*] Simulating Request with Ticked IDs: {args.tests}")
    implied = kg.implied_tests(args.tests)
    print(f"    Implied individual tests: {sorted(implied)}")
    expected_tubes = kg.calculate_expected_tubes(args.tests)
    print(f"    Required specimen tubes: {expected_tubes}")

    observed_tubes: dict[str, int | None] = {}
    for item in args.tubes:
        if ":" in item:
            name, count_str = item.split(":", 1)
            try:
                observed_tubes[name] = int(count_str)
            except ValueError:
                observed_tubes[name] = None

    print(f"    Observed physical tubes: {observed_tubes}")

    report = kg.validate_request(ticked_ids=args.tests, observed_tubes=observed_tubes)
    print("\n[+] Validation Report:")
    print(f"    Status: {'VALID' if report.is_valid else 'DISCREPANCY DETECTED'}")
    print(f"    Confidence: {report.confidence:.2f}")

    if report.discrepancies:
        print("    Discrepancies:")
        for d in report.discrepancies:
            print(f"      - [FAIL] {d}")
    else:
        print("    Discrepancies: None")

    if report.warnings:
        print("    Warnings:")
        for w in report.warnings:
            print(f"      - [WARN] {w}")
    else:
        print("    Warnings: None")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

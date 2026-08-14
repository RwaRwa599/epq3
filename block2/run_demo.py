#!/usr/bin/env python3
"""Interactive CLI demo for Block 2: Clinical Knowledge Graph & Prior Engine."""

import argparse
import json
import sys
from pathlib import Path

# Add local directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent))

from med_doc.kg import KnowledgeGraph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block 2 CLI Demo — Clinical Knowledge Graph & Prior Engine"
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
        help="List of ticked test or profile IDs to simulate",
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
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save validation JSON report",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  BLOCK 2: CLINICAL KNOWLEDGE GRAPH & PRIOR VALIDATION ENGINE")
    print("=" * 70)

    # 1. Load Knowledge Graph
    kg = KnowledgeGraph.load(args.kg_path)
    print(f"\n[+] Loaded Knowledge Graph: version='{kg.version}' ({len(kg.catalogue)} catalogue tests)")
    print(f"    Disclaimer: {kg.disclaimer}\n")

    # 2. Resolve alias
    resolved_id = kg.resolve_alias(args.resolve)
    print(f"[*] Alias Resolution: '{args.resolve}' -> '{resolved_id}'")

    # 3. Fuzzy match doctor handwriting
    fuzzy_candidates = kg.fuzzy_match_catalogue(args.fuzzy, top_k=3)
    print(f"[*] Fuzzy Match for write-in '{args.fuzzy}':")
    for rank, c in enumerate(fuzzy_candidates, 1):
        print(f"    {rank}. {c.value} (id: {c.canonical_id}) - Score: {c.score:.3f} [Tier {c.tier}]")

    # 4. Expand profiles & calculate expected tubes
    print(f"\n[*] Simulating Request with Ticked IDs: {args.tests}")
    implied = kg.implied_tests(args.tests)
    print(f"    Implied individual tests: {sorted(implied)}")
    expected_tubes = kg.calculate_expected_tubes(args.tests)
    print(f"    Required specimen tubes: {expected_tubes}")

    # 5. Parse observed tubes
    observed_tubes: dict[str, int | None] = {}
    for item in args.tubes:
        if ":" in item:
            name, count_str = item.split(":", 1)
            try:
                observed_tubes[name] = int(count_str)
            except ValueError:
                observed_tubes[name] = None

    print(f"    Observed physical tubes: {observed_tubes}")

    # 6. Run clinical validation
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

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(report.model_dump_json(indent=2))
        print(f"\n[+] Saved validation report to {out_p}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

"""Standalone CLI demonstration of Block 1 Document Normalization with Batch & ZIP support."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import cv2
from PIL import Image

# Ensure local med_doc is importable
sys.path.insert(0, str(Path(__file__).parent))

from med_doc.normalization.batch import normalize_batch
from med_doc.normalization.pipeline import normalize_document
from med_doc.paths import DEFAULT_TEMPLATE, OUTPUTS_DIR, SAMPLES_DIR, V1_TEMPLATE


def main():
    parser = argparse.ArgumentParser(description="Run Block 1 Normalization (single image or batch ZIP).")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help="Path to input image, folder of images, or ZIP of images",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR / "block1_batch"),
        help="Output directory for crops and overlays",
    )
    parser.add_argument(
        "--output-zip",
        default=str(OUTPUTS_DIR / "block1_normalized_batch.zip"),
        help="Path to save output ZIP file for Block 2",
    )
    parser.add_argument(
        "--template",
        choices=["auto", "v0", "v1"],
        default="auto",
        help="Template revision to use",
    )
    args = parser.parse_args()

    if args.input_path is None:
        # Search for sample images
        samples = list(SAMPLES_DIR.glob("*.png")) + list(SAMPLES_DIR.glob("*.jpg"))
        if not samples:
            print("Error: No input image provided and no samples found in samples/")
            sys.exit(1)
        input_target = samples
        print(f"Using default sample images ({len(samples)} found): {[s.name for s in samples]}")
    else:
        input_target = Path(args.input_path)
        if not input_target.exists():
            print(f"Error: Path not found: {input_target}")
            sys.exit(1)

    template_arg = None
    if args.template == "v0":
        template_arg = DEFAULT_TEMPLATE
    elif args.template == "v1":
        template_arg = V1_TEMPLATE

    print("=" * 70)
    print("  BLOCK 1: DOCUMENT NORMALIZATION & BATCH ROI EXTRACTION")
    print("=" * 70)

    # Run batch normalization
    result = normalize_batch(
        input_target,
        output_dir=args.output_dir,
        output_zip=args.output_zip,
        template=template_arg,
    )

    manifest = result["manifest"]
    print("\n" + "=" * 70)
    print(f"✓ Batch normalization finished!")
    print(f"  Total Documents:      {manifest['total_documents']}")
    print(f"  Successful:           {manifest['successful_documents']}")
    print(f"  Output Directory:     {result['output_dir']}")
    if result["output_zip"]:
        print(f"  Output ZIP (Block 2): {result['output_zip']}")
    print("=" * 70)


if __name__ == "__main__":
    main()

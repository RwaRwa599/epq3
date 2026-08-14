"""Standalone CLI demonstration of Block 1 Document Normalization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import cv2
from PIL import Image

# Ensure local med_doc is importable
sys.path.insert(0, str(Path(__file__).parent))

from med_doc.normalization.pipeline import normalize_document
from med_doc.paths import OUTPUTS_DIR, SAMPLES_DIR, DEFAULT_TEMPLATE, V1_TEMPLATE


def main():
    parser = argparse.ArgumentParser(description="Run Block 1 Normalization on an image.")
    parser.add_argument("input_image", nargs="?", default=None, help="Path to input medical document image")
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR), help="Output directory for crops and overlays")
    parser.add_argument("--template", choices=["auto", "v0", "v1"], default="auto", help="Template revision to use")
    args = parser.parse_args()

    if args.input_image is None:
        # Search for a sample image
        samples = list(SAMPLES_DIR.glob("*.png")) + list(SAMPLES_DIR.glob("*.jpg"))
        if not samples:
            print("Error: No input image provided and no samples found in samples/")
            sys.exit(1)
        input_path = samples[0]
        print(f"Using default sample image: {input_path}")
    else:
        input_path = Path(args.input_image)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    template_arg = None
    if args.template == "v0":
        template_arg = DEFAULT_TEMPLATE
    elif args.template == "v1":
        template_arg = V1_TEMPLATE

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Processing: {input_path.name} ---")
    img = Image.open(input_path)
    print(f"Original size: {img.size[0]}x{img.size[1]} | Mode: {img.mode}")

    result = normalize_document(img, template=template_arg, document_id=input_path.stem)

    print(f"Template chosen: {result.extra.get('template_id')}")
    print(f"Warp method: {result.warp_method} | Orientation: {result.orientation_degrees}°")
    print(f"Alignment confidence: {result.alignment_confidence:.3f}")
    print(f"Extracted crops: {len(result.checkbox_crops)} checkboxes, {len(result.handwriting_crops)} handwriting fields")

    # Save visual overlay
    overlay_path = out_dir / f"{input_path.stem}_overlay.jpg"
    if result.debug_overlay is not None:
        cv2.imwrite(str(overlay_path), cv2.cvtColor(result.debug_overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"Saved visual overlay: {overlay_path}")

    # Save sample crops
    sample_crops_dir = out_dir / f"{input_path.stem}_crops"
    sample_crops_dir.mkdir(parents=True, exist_ok=True)

    for fid, crop in list(result.checkbox_crops.items()) + list(result.handwriting_crops.items()):
        raw_bgr = cv2.cvtColor(crop.raw_image, cv2.COLOR_RGB2BGR) if crop.raw_image.ndim == 3 else crop.raw_image
        norm_bgr = cv2.cvtColor(crop.normalized_image, cv2.COLOR_RGB2BGR) if crop.normalized_image.ndim == 3 else crop.normalized_image
        cv2.imwrite(str(sample_crops_dir / f"{fid}_raw.png"), raw_bgr)
        cv2.imwrite(str(sample_crops_dir / f"{fid}_norm.png"), norm_bgr)

    print(f"Saved all {len(result.checkbox_crops) + len(result.handwriting_crops)} crops to: {sample_crops_dir}")
    print("\n✓ Normalization complete!")


if __name__ == "__main__":
    main()

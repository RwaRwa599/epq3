"""Measure checkbox overlay offset on a local canonical.png or a metadata dump.

Clinic photos stay gitignored. Typical inputs:

  data/samples/private/canonical.png
  data/samples/private/metadata.json

Prints template vs post-snap vs 1c-override centers so a translation can be
separated from snap_overlay pulling toward letter-loops.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from med_doc.normalization.align import snap_overlay
from med_doc.normalization.detect import detect_checkboxes_photo, filter_checkbox_boxes
from med_doc.paths import PRIVATE_SAMPLES_DIR, V1_TEMPLATE
from med_doc.template import load_template

PROBE_IDS = ("body_check_plan_1", "hba1c", "creatinine", "na", "cbc")


def _center(bbox: list[float] | list[int]) -> tuple[float, float]:
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def _px_center(template, spec) -> tuple[float, float]:
    x0, y0, x1, y1 = template.pixel_bbox(spec, apply_pad=False)
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def summarize_offsets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dxs = [r["dx"] for r in rows if r.get("dx") is not None]
    dys = [r["dy"] for r in rows if r.get("dy") is not None]
    out: dict[str, Any] = {
        "n": len(rows),
        "median_dx": round(float(np.median(dxs)), 2) if dxs else None,
        "median_dy": round(float(np.median(dys)), 2) if dys else None,
        "dx_range": [round(min(dxs), 2), round(max(dxs), 2)] if dxs else None,
        "dy_range": [round(min(dys), 2), round(max(dys), 2)] if dys else None,
    }
    if dxs and max(abs(x) for x in dxs) > 1e-6:
        # Growing with Y ⇒ scale; flat across Y ⇒ translation.
        ys = np.array([r["template_cy"] for r in rows if r.get("dx") is not None], dtype=float)
        xs = np.array(dxs, dtype=float)
        if len(xs) >= 3 and float(np.std(ys)) > 1.0:
            slope = float(np.polyfit(ys, xs, 1)[0])
            out["dx_vs_y_slope"] = round(slope, 5)
            out["kind"] = "scale" if abs(slope) > 0.02 else "translation"
        else:
            out["kind"] = "translation"
    return out


def measure_from_metadata(meta: dict[str, Any], *, template_path: Path | None = None) -> dict[str, Any]:
    template = load_template(template_path or V1_TEMPLATE)
    by_id = template.field_map()
    fields = (meta.get("fields") or {}).get("checkboxes") or {}
    overrides = ((meta.get("extra") or {}).get("bbox_overrides")) or meta.get("bbox_overrides") or {}
    w, h = template.width, template.height
    rows: list[dict[str, Any]] = []
    ok_rows: list[dict[str, Any]] = []
    retry_rows: list[dict[str, Any]] = []
    for fid, spec in by_id.items():
        if spec.field_type != "checkbox":
            continue
        rec = fields.get(fid) or {}
        bbox = rec.get("bbox")
        if not bbox:
            continue
        tcx, tcy = _px_center(template, spec)
        ocx, ocy = _center(bbox)
        row = {
            "field_id": fid,
            "status": rec.get("crop_validate_status"),
            "attempts": rec.get("crop_validate_attempts"),
            "template_cx": round(tcx, 1),
            "template_cy": round(tcy, 1),
            "obs_cx": round(ocx, 1),
            "obs_cy": round(ocy, 1),
            "dx": round(ocx - tcx, 2),
            "dy": round(ocy - tcy, 2),
            "rel_dx": round((ocx - tcx) / w, 5),
            "rel_dy": round((ocy - tcy) / h, 5),
        }
        ov = overrides.get(fid)
        if ov:
            rcx, rcy = _center(ov)
            row["override_cx"] = round(rcx, 1)
            row["override_cy"] = round(rcy, 1)
            row["override_dx"] = round(rcx - tcx, 2)
            row["override_dy"] = round(rcy - tcy, 2)
        rows.append(row)
        if rec.get("crop_validate_status") == "ok" and int(rec.get("crop_validate_attempts") or 0) == 0:
            ok_rows.append(row)
        elif rec.get("crop_validate_status") == "retry":
            retry_rows.append(row)
    probes = [r for r in rows if r["field_id"] in PROBE_IDS]
    return {
        "source": "metadata",
        "template_id": template.template_id,
        "canvas_size": [w, h],
        "probes": probes,
        "first_try_ok": summarize_offsets(ok_rows),
        "retry_overrides": summarize_offsets(retry_rows),
        "all_observed": summarize_offsets(rows),
        "note": (
            "retry/override ≈ printed square after 1c; first-try ok ≈ 1b window "
            "(glyph if hollow-ring was too loose). Constant dx ⇒ JSON translation; "
            "ok-dx ≫ override-dx ⇒ snap_overlay, do not rewrite JSON."
        ),
    }


def measure_from_canonical(path: Path, *, template_path: Path | None = None) -> dict[str, Any]:
    import cv2

    template = load_template(template_path or V1_TEMPLATE)
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    canvas = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raw = detect_checkboxes_photo(canvas)
    filtered = filter_checkbox_boxes(canvas, raw)
    snapped, snap_meta = snap_overlay(canvas, template)
    by_id = snapped.field_map()
    orig = template.field_map()
    h, w = canvas.shape[:2]
    rows = []
    for fid in PROBE_IDS:
        spec = orig.get(fid)
        if spec is None:
            continue
        tcx, tcy = _px_center(template, spec)
        scx, scy = _px_center(snapped, by_id[fid])
        y = int(round(tcy))
        strip = canvas[max(0, y - 20) : min(h, y + 20), 0 : min(w, int(tcx) + 80)]
        gutter = detect_checkboxes_photo(strip)
        printed = None
        if gutter:
            gx0 = float(gutter[0][0])
            gy0 = float(gutter[0][1]) + max(0, y - 20)
            printed = (gx0 + (gutter[0][2] - gutter[0][0]) / 2.0, gy0 + (gutter[0][3] - gutter[0][1]) / 2.0)
        row = {
            "field_id": fid,
            "template_cx": round(tcx, 1),
            "template_cy": round(tcy, 1),
            "snapped_cx": round(scx, 1),
            "snapped_cy": round(scy, 1),
            "dx": round(scx - tcx, 2),
            "dy": round(scy - tcy, 2),
            "printed_gutter": None if printed is None else [round(printed[0], 1), round(printed[1], 1)],
            "printed_dx": None if printed is None else round(printed[0] - tcx, 2),
        }
        rows.append(row)
    return {
        "source": str(path),
        "n_detected": len(raw),
        "n_filtered": len(filtered),
        "snap": {k: snap_meta.get(k) for k in ("dx", "dy", "n_snapped", "n_detected", "method", "confidence")},
        "probes": rows,
        "summary": summarize_offsets(rows),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical", type=Path, default=PRIVATE_SAMPLES_DIR / "canonical.png")
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--template", type=Path, default=V1_TEMPLATE)
    args = p.parse_args(argv)
    if args.metadata and args.metadata.is_file():
        meta = json.loads(args.metadata.read_text(encoding="utf-8"))
        report = measure_from_metadata(meta, template_path=args.template)
    elif args.canonical.is_file():
        report = measure_from_canonical(args.canonical, template_path=args.template)
    else:
        p.error(
            f"need --metadata JSON or a canonical.png (looked for {args.canonical})"
        )
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Crop PHI strips off lab photos (geometry crop, not redaction).

Default backend is the canonical template band (drop header + office footer).
Optional LayoutParser: keep PubLayNet types (Table/Text/List), drop Title.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

from med_doc.normalization.inputs import collect_image_inputs
from med_doc.paths import ROOT

Backend = Literal["template", "layoutparser", "layoutparser_then_template"]
Combine = Literal["union", "largest", "vertical_span"]
Engine = Literal["paddle", "detectron2"]

DEFAULT_CONFIG = ROOT / "configs" / "layout_crop.json"
PUBLAYNET_LABELS = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}


class TemplateBand(BaseModel):
    """Keep this fraction of the page. 0–1 relative to image width/height."""

    top: float = Field(ge=0.0, le=1.0, default=0.10)
    bottom: float = Field(ge=0.0, le=1.0, default=0.88)
    left: float = Field(ge=0.0, le=1.0, default=0.0)
    right: float = Field(ge=0.0, le=1.0, default=1.0)


class LayoutParserSpec(BaseModel):
    engine: Engine = "paddle"
    config_path: str = "lp://PubLayNet/ppyolov2_r50vd_dcn_365e"
    detectron2_config_path: str = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
    label_map: dict[str, str] = Field(
        default_factory=lambda: {str(k): v for k, v in PUBLAYNET_LABELS.items()}
    )
    enforce_cpu: bool = True


class CropConfig(BaseModel):
    """What to keep for a whole batch. Load from configs/layout_crop.json."""

    backend: Backend = "template"
    keep_types: list[str] = Field(default_factory=lambda: ["Table", "Text", "List"])
    drop_types: list[str] = Field(default_factory=lambda: ["Title", "Figure"])
    combine: Combine = "vertical_span"
    score_threshold: float = Field(ge=0.0, le=1.0, default=0.5)
    min_area_frac: float = Field(ge=0.0, le=1.0, default=0.08)
    padding_px: int = Field(ge=0, default=0)
    layoutparser: LayoutParserSpec = Field(default_factory=LayoutParserSpec)
    template: TemplateBand = Field(default_factory=TemplateBand)


def load_crop_config(path: str | Path | None = None) -> CropConfig:
    if path is None:
        path = DEFAULT_CONFIG
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    tmpl = raw.get("template")
    if isinstance(tmpl, dict):
        raw["template"] = {k: v for k, v in tmpl.items() if k != "comment"}
    return CropConfig.model_validate(raw)


def _label_map(spec: LayoutParserSpec) -> dict[int, str]:
    out: dict[int, str] = {}
    for k, v in spec.label_map.items():
        out[int(k)] = v
    return out or dict(PUBLAYNET_LABELS)


def _xyxy(block: Any) -> tuple[int, int, int, int]:
    r = getattr(block, "block", block)
    return int(r.x_1), int(r.y_1), int(r.x_2), int(r.y_2)


def _area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _pad(box: tuple[int, int, int, int], pad: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    )


def template_box(h: int, w: int, band: TemplateBand) -> tuple[int, int, int, int]:
    if band.bottom <= band.top or band.right <= band.left:
        raise ValueError("template crop band is empty: check top < bottom and left < right")
    return (
        int(band.left * w),
        int(band.top * h),
        int(band.right * w),
        int(band.bottom * h),
    )


def combine_boxes(
    boxes: list[tuple[int, int, int, int]],
    *,
    combine: Combine,
    w: int,
    h: int,
) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    if combine == "largest":
        return max(boxes, key=_area)
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    if combine == "vertical_span":
        return 0, y1, w, y2
    return x1, y1, x2, y2


def boxes_from_layout(layout: Any, cfg: CropConfig, *, w: int, h: int) -> list[tuple[int, int, int, int]]:
    keep = {t.lower() for t in cfg.keep_types}
    drop = {t.lower() for t in cfg.drop_types}
    out: list[tuple[int, int, int, int]] = []
    for block in layout or []:
        kind = str(getattr(block, "type", "") or "").strip()
        score = float(getattr(block, "score", 1.0) or 1.0)
        if score < cfg.score_threshold:
            continue
        if kind.lower() in drop:
            continue
        if keep and kind.lower() not in keep:
            continue
        box = _xyxy(block)
        if _area(box) < cfg.min_area_frac * w * h:
            continue
        out.append(box)
    return out


def load_layoutparser_model(cfg: CropConfig):
    try:
        import layoutparser as lp
    except ImportError as exc:
        raise ImportError(
            "layoutparser is not installed. pip install layoutparser "
            "and either paddlepaddle (engine=paddle) or detectron2 "
            "(engine=detectron2). Template backend needs neither."
        ) from exc

    spec = cfg.layoutparser
    labels = _label_map(spec)
    if spec.engine == "paddle":
        model_cls = getattr(lp, "PaddleDetectionLayoutModel", None)
        if model_cls is None:
            raise ImportError("This layoutparser build has no PaddleDetectionLayoutModel")
        return model_cls(
            config_path=spec.config_path,
            threshold=cfg.score_threshold,
            label_map=labels,
            enforce_cpu=spec.enforce_cpu,
        )
    extra = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", cfg.score_threshold]
    return lp.Detectron2LayoutModel(
        spec.detectron2_config_path,
        extra_config=extra,
        label_map=labels,
    )


def _bgr(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        img = image
        if img.ndim == 3 and img.shape[2] == 3:
            return img
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    path = Path(image)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def crop_array(
    image: np.ndarray,
    cfg: CropConfig,
    *,
    model: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = image.shape[:2]
    used = cfg.backend
    box: tuple[int, int, int, int] | None = None
    n_blocks = 0

    if cfg.backend in ("layoutparser", "layoutparser_then_template"):
        if model is None:
            model = load_layoutparser_model(cfg)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        layout = model.detect(rgb)
        n_blocks = len(list(layout or []))
        kept = boxes_from_layout(layout, cfg, w=w, h=h)
        box = combine_boxes(kept, combine=cfg.combine, w=w, h=h)
        if box is None and cfg.backend == "layoutparser":
            raise RuntimeError(
                "LayoutParser kept no blocks. Set keep_types to match detected "
                "labels, lower score_threshold, or use backend=layoutparser_then_template."
            )
        if box is None:
            used = "template"

    if box is None:
        box = template_box(h, w, cfg.template)
        used = "template"

    box = _pad(box, cfg.padding_px, w, h)
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError(f"Empty crop box {box} on {w}x{h}")
    return crop, {
        "backend_used": used,
        "bbox_xyxy": [x1, y1, x2, y2],
        "n_layout_blocks": n_blocks,
        "keep_types": list(cfg.keep_types),
        "drop_types": list(cfg.drop_types),
        "combine": cfg.combine,
    }


def crop_batch(
    inputs: str | Path | list,
    output_dir: str | Path,
    *,
    config: str | Path | CropConfig | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    """Crop a folder, ZIP, file, or list of images. Writes PNGs + manifest.json."""
    cfg = config if isinstance(config, CropConfig) else load_crop_config(config)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    items = collect_image_inputs(inputs)
    loaded_model = model
    if cfg.backend.startswith("layoutparser") and loaded_model is None:
        loaded_model = load_layoutparser_model(cfg)

    documents: list[dict[str, Any]] = []
    for doc_id, item in items:
        try:
            bgr = _bgr(item)
            cropped, meta = crop_array(bgr, cfg, model=loaded_model)
            out_name = f"{doc_id}.png"
            out_path = target / out_name
            cv2.imwrite(str(out_path), cropped)
            documents.append({"doc_id": doc_id, "status": "success", "path": out_name, **meta})
        except Exception as exc:
            documents.append({"doc_id": doc_id, "status": "error", "error": str(exc)})

    manifest = {
        "version": "1.0",
        "stage": "layout_crop",
        "backend": cfg.backend,
        "keep_types": cfg.keep_types,
        "drop_types": cfg.drop_types,
        "combine": cfg.combine,
        "template": cfg.template.model_dump(),
        "total": len(items),
        "successful": sum(1 for d in documents if d.get("status") == "success"),
        "documents": documents,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (target / "crop_config_used.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    return {"manifest": manifest, "output_dir": str(target)}

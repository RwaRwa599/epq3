"""Layout crop for PHI strips (template band + optional LayoutParser)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from med_doc.privacy.layout_crop import (
    CropConfig,
    TemplateBand,
    boxes_from_layout,
    combine_boxes,
    crop_array,
    crop_batch,
    load_crop_config,
    template_box,
)


def test_default_config_loads():
    cfg = load_crop_config()
    assert cfg.backend == "template"
    assert "Table" in cfg.keep_types
    assert "Title" in cfg.drop_types
    assert cfg.template.top == 0.08
    assert cfg.template.bottom == 0.82


def test_template_crop_keeps_clinical_info_drops_name_header():
    h, w = 1000, 80
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:80] = (0, 0, 255)  # name header ~0–0.08
    img[80:99] = (0, 255, 255)  # clinical_info starts just under the header bar
    img[100:820] = (255, 0, 0)  # columns
    img[820:] = (0, 255, 0)  # footer
    crop, meta = crop_array(img, CropConfig())
    assert meta["backend_used"] == "template"
    y1, y2 = meta["bbox_xyxy"][1], meta["bbox_xyxy"][3]
    assert y1 == 80
    assert y2 == 820
    assert (crop[0, 0] == (0, 255, 255)).all()
    assert (crop[15, 0] == (0, 255, 255)).all()
    assert (crop[40, 0] == (255, 0, 0)).all()
    assert (crop[-1, 0] == (255, 0, 0)).all()


def test_layoutparser_keep_types_vertical_span():
    h, w = 200, 100
    title = SimpleNamespace(
        type="Title",
        score=0.99,
        block=SimpleNamespace(x_1=0, y_1=0, x_2=w, y_2=20),
    )
    table = SimpleNamespace(
        type="Table",
        score=0.95,
        block=SimpleNamespace(x_1=5, y_1=30, x_2=90, y_2=160),
    )
    cfg = CropConfig(keep_types=["Table", "Text"], drop_types=["Title"], min_area_frac=0.01)
    kept = boxes_from_layout([title, table], cfg, w=w, h=h)
    assert kept == [(5, 30, 90, 160)]
    box = combine_boxes(kept, combine="vertical_span", w=w, h=h)
    assert box == (0, 30, 100, 160)


def test_layoutparser_then_template_fallback():
    img = np.ones((100, 80, 3), dtype=np.uint8) * 7

    class EmptyModel:
        def detect(self, _rgb):
            return []

    cfg = CropConfig(backend="layoutparser_then_template", template=TemplateBand(top=0.2, bottom=0.7))
    crop, meta = crop_array(img, cfg, model=EmptyModel())
    assert meta["backend_used"] == "template"
    assert crop.shape[0] == 50


def test_crop_batch_writes_pngs(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    img = np.zeros((40, 30, 3), dtype=np.uint8)
    img[4:36] = 200
    import cv2

    cv2.imwrite(str(src / "a.png"), img)
    cv2.imwrite(str(src / "b.png"), img)
    out = tmp_path / "out"
    result = crop_batch(src, out, config=CropConfig(template=TemplateBand(top=0.1, bottom=0.9)))
    assert result["manifest"]["successful"] == 2
    assert (out / "a.png").is_file()
    assert (out / "manifest.json").is_file()

    dbg = crop_batch(
        src,
        tmp_path / "dbg",
        config=CropConfig(template=TemplateBand(top=0.1, bottom=0.9)),
        debug=True,
    )
    assert (tmp_path / "dbg" / "debug" / "a_boxes.png").is_file()
    assert dbg["manifest"]["documents"][0]["debug_path"] == "debug/a_boxes.png"

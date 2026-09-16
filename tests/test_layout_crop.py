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
    assert cfg.template.bottom == 1.0


def test_template_crop_keeps_clinical_info_and_footer():
    h, w = 1000, 80
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:80] = (0, 0, 255)  # name header ~0–0.08
    img[80:99] = (0, 255, 255)  # clinical_info
    img[100:820] = (255, 0, 0)  # columns
    img[820:] = (0, 255, 0)  # tubes / office — keep
    crop, meta = crop_array(img, CropConfig())
    assert meta["backend_used"] == "template"
    y1, y2 = meta["bbox_xyxy"][1], meta["bbox_xyxy"][3]
    assert y1 == 80
    assert y2 == 1000
    assert (crop[0, 0] == (0, 255, 255)).all()
    assert (crop[15, 0] == (0, 255, 255)).all()
    assert (crop[40, 0] == (255, 0, 0)).all()
    assert (crop[-1, 0] == (0, 255, 0)).all()


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


def test_standalone_crop_script(tmp_path):
    import subprocess
    import sys

    from med_doc.paths import ROOT
    from PIL import Image

    src = tmp_path / "in"
    src.mkdir()
    Image.new("RGB", (100, 200), (10, 20, 30)).save(src / "page.png")
    out = tmp_path / "out"
    script = ROOT / "scripts" / "crop_labform_pages.py"
    subprocess.check_call(
        [sys.executable, str(script), str(src), "--out", str(out), "--debug"],
    )
    cropped = Image.open(out / "page.png")
    assert cropped.size[0] == 100
    assert cropped.size[1] < 200
    assert cropped.size[1] == 200 - int(round(200 * (10.5 / 29.7)))
    assert (out / "debug" / "page_boxes.png").is_file()
    assert (out / "crop_eval.json").is_file()


def _form_page(h: int = 1200, w: int = 800, *, info_y: int, header_y: int):
    import cv2
    import numpy as np

    img = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.putText(img, "Patient Name PHI", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2)
    cv2.putText(img, "Clinical Information", (30, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 1)
    cv2.rectangle(img, (20, header_y - 22), (w - 20, header_y + 8), (35, 35, 35), -1)
    cv2.putText(img, "HAEMATOLOGY", (36, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 1)
    return img


def test_ten_cm_crop_does_not_cut_clinical_info_or_column_headers(tmp_path):
    import importlib.util
    import sys

    from med_doc.paths import ROOT
    from PIL import Image

    spec = importlib.util.spec_from_file_location(
        "crop_labform_pages", ROOT / "scripts" / "crop_labform_pages.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys

    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    h, w = 1200, 800
    proposed = mod.top_px_from_cm(h, 10.5, dpi=None)
    assert 400 < proposed < 450  # 10.5/29.7 of A4
    bgr = _form_page(h, w, info_y=180, header_y=250)
    rgb = bgr
    im = Image.fromarray(rgb)
    plan = mod.plan_crop(im, top_cm=10.5)
    assert plan["adjusted"] is True
    assert plan["top_px"] < proposed
    assert plan["top_px"] <= 180
    crop = im.crop((0, plan["top_px"], w, h))
    ev = mod.evaluate_crop(crop)
    assert ev["clinical_information"] is True, ev
    assert ev["column_headers"] is True, ev
    assert ev["ok"] is True

    safe = _form_page(h, w, info_y=520, header_y=600)
    plan_ok = mod.plan_crop(Image.fromarray(safe), top_cm=10.5)
    assert plan_ok["top_px"] == proposed
    crop_ok = Image.fromarray(safe).crop((0, plan_ok["top_px"], w, h))
    ev_ok = mod.evaluate_crop(crop_ok)
    assert ev_ok["ok"] is True, ev_ok

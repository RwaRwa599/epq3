# Layout crop — batch PHI strip (not redaction)

Crops the **header and office footer off the page**. Pixels are removed, not blacked out. Config for a whole batch: [`configs/layout_crop.json`](../../configs/layout_crop.json).

**Colab:** [Layout_Crop_Batch.ipynb](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Layout_Crop_Batch.ipynb)

Do this **locally** if the goal is not uploading names. Then point `run_blocks_1_to_5` at the cropped folder.

## Set the crop for a batch

Edit `configs/layout_crop.json` (or pass CLI flags). One file applies to every image in the folder/ZIP.

| Field | What it does |
|---|---|
| `backend` | `template` (default) · `layoutparser` · `layoutparser_then_template` |
| `template.top` / `.bottom` / `.left` / `.right` | Keep this **fraction of the page** (0–1). Default **0.10–0.88** drops v1 `header_bar` (~0–0.08) and footer (~0.90–1). |
| `keep_types` | LayoutParser PubLayNet labels to **keep**: `Table`, `Text`, `List` |
| `drop_types` | Labels to **throw away**: `Title`, `Figure` |
| `combine` | How to merge kept boxes: `vertical_span` (full width, min–max Y — best for “cut header/footer”), `union`, `largest` |
| `score_threshold` | Ignore LayoutParser boxes below this |
| `min_area_frac` | Ignore tiny boxes |
| `padding_px` | Extra pixels around the kept box (0 for template so the band stays exact) |

## Template (no extra packages)

```bash
python -m med_doc.privacy photos/ --out cropped/
python -m med_doc.privacy photos.zip --out cropped/ --top 0.12 --bottom 0.86
```

```python
from med_doc.privacy import crop_batch, CropConfig, load_crop_config

cfg = load_crop_config()           # configs/layout_crop.json
cfg.template.top = 0.12            # cut more header
cfg.template.bottom = 0.86         # cut more footer
crop_batch("photos/", "cropped/", config=cfg)
```

Then `run_blocks_1_to_5("cropped/", output_dir="pipeline_out", output_mode="dev")`.

## LayoutParser ([Layout-Parser/layout-parser](https://github.com/Layout-Parser/layout-parser))

PubLayNet types: **Text, Title, List, Table, Figure**. For this form the checkbox grid is usually `Table` or `Text`; the patient header is often `Title`. Keep table/text, drop title, combine with `vertical_span`.

```bash
pip install layoutparser paddlepaddle
python -m med_doc.privacy photos/ --out cropped/ \
  --backend layoutparser_then_template \
  --keep Table Text List --drop Title Figure \
  --combine vertical_span
```

`layoutparser_then_template` falls back to the JSON band if LayoutParser finds nothing (typical on phone photos of this sheet). Detectron2 (`layoutparser.engine: detectron2`) is optional and heavier.

```python
from med_doc.privacy import CropConfig, crop_batch

cfg = CropConfig(
    backend="layoutparser_then_template",
    keep_types=["Table", "Text", "List"],
    drop_types=["Title", "Figure"],
    combine="vertical_span",
)
crop_batch("photos/", "cropped/", config=cfg)
```

This is **not** NER de-identification. Write-in `others` in the grid stays in the crop if it sits in the kept band.

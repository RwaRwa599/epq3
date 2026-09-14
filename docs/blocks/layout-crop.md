# Layout crop — local batch with LayoutParser

Crops **header/footer pixels off the page**. Not redaction. One config applies to **every** image in a folder.

- Default band (no extra packages): [`configs/layout_crop.json`](../../configs/layout_crop.json) — `backend: template`
- LayoutParser preset: [`configs/layout_crop.layoutparser.json`](../../configs/layout_crop.layoutparser.json)

PubLayNet labels (this is the layout you set): **Text, Title, List, Table, Figure**.

| You set | Effect |
|---|---|
| `keep_types` | Boxes **kept**. Columns look like `Table` / `List`. **Do not keep `Text`** — the patient header is usually Text and `vertical_span` would pull it in. |
| `drop_types` | Boxes **thrown away**: `Title`, `Figure`, `Text`. |
| `combine` | `largest` = the biggest kept box (the column grid). Avoid `vertical_span` if a header Text box is kept. |
| `template.top` / `.bottom` | Column band if LayoutParser finds nothing. Default **0.10–0.82** (v1 gutters). Header ~0–0.08 is cropped off. |

Green / red / orange on `--debug` overlays: **keep / drop / final crop**.

## 1. Install (once, local)

```bash
pip install -e ".[layoutparser]"
pip install paddlepaddle
```

Detectron2 is optional. The preset uses LayoutParser’s **Paddle** PubLayNet model (`lp://PubLayNet/ppyolov2_r50vd_dcn_365e`).

## 2. Point at a folder and set the layout

Edit `configs/layout_crop.layoutparser.json` (keep/drop/combine), then:

```bash
python -m med_doc.privacy /path/to/photos --out cropped/ \
  --config configs/layout_crop.layoutparser.json \
  --debug
```

Or override without editing JSON:

```bash
python -m med_doc.privacy /path/to/photos --out cropped/ \
  --backend layoutparser_then_template \
  --keep Table List \
  --drop Title Figure Text \
  --combine largest \
  --score 0.5 \
  --debug
```

`--debug` writes `cropped/debug/<id>_boxes.png`. Open those first. If the orange box still includes the name header, add that label to `--drop` (often `Title`). If the test grid is missing, add the printed type from `detected_types` in `cropped/manifest.json` to `--keep`, or lower `--score`.

## 3. If LayoutParser misses the grid

`layoutparser_then_template` falls back to the JSON band (`top`/`bottom`). Tune that instead:

```bash
python -m med_doc.privacy /path/to/photos --out cropped/ --top 0.10 --bottom 0.82 --debug
```

That needs **no** LayoutParser install.

## 4. Then run Blocks 1–5 on the cropped folder

```bash
python -c "from med_doc import run_blocks_1_to_5; run_blocks_1_to_5('cropped/', output_dir='pipeline_out', output_mode='dev')"
```

Do this **before** Colab if you do not want names uploaded. Write-in `others` inside the grid stays in the crop.

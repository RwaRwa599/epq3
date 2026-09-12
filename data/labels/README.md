# Crop-level tick labels

Gold is **field ids**, not page PNGs.

```json
{
  "version": "1.0",
  "kind": "checkbox_ticks",
  "sheets": [
    {"doc_id": "IMG_7596", "ticked_field_ids": ["ca125"], "notes": "only CA 125"}
  ]
}
```

Clinic photos and exported crops stay **local** (`data/labels/crops/`, `data/samples/private/`). Do not git-add them.

```bash
# 1. Pipeline in dev mode → block1.zip
python -c "from med_doc import run_blocks_1_to_5; run_blocks_1_to_5('sheet.jpg', output_dir='out', output_mode='dev')"

# 2. Split checkbox PNGs using gold
python -m med_doc.eval export --block1 out/block1.zip --gold data/labels/examples/IMG_7596.json --out data/labels/crops

# 3. Refit logistic (synthetic empties + clinic empties heavily weighted)
python -m med_doc.eval train --crops data/labels/crops --write data/labels/mark_weights.json

# 4. If CA 125 is still FN: is the crop a box or a label?
python -m med_doc.eval crop-qa --block1 out/block1.zip --gold data/labels/examples/IMG_7596.json --field ca125

python -m med_doc.eval models
```

Ticks use **geometry + logreg**, not Paddle/TrOCR. A crop CNN is not wired until logreg plateaus.

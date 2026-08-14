# Block 3 — Handwriting Recognition & Prior Fusion Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block3/block3/Block_3_Handwriting_Recognition.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Block 3 classifies checkbox marks, reads handwriting crops, and fuses those drafts with Block 2 clinical priors. It ingests `block2_validated_batch.zip` and exports `block3_predictions_batch.zip` for Block 4 / LIS.

---

## Batch pipeline

```
[ block2_validated_batch.zip from Block 2 ]
                     |
                     v  (Colab upload popup)
+--------------------------------------------------------+
| BLOCK 3                                                |
| * Checkbox mark classifier (density / slash / filled)  |
| * Digit, date, and optional TrOCR handwriting readers  |
| * Prior fusion against kg.assume() rankings            |
| * HiTL flags for borderline marks and low-confidence   |
+--------------------------------------------------------+
                     |
                     v
[ block3_predictions_batch.zip for Block 4 ]
```

---

## Quick start

```bash
cd block3
pip install -r requirements.txt
python run_demo.py --input-zip block2_validated_batch.zip --output-zip block3_predictions_batch.zip
pytest -v
```

TrOCR is optional (`pip install transformers torch`). The default `auto` backend uses TrOCR when installed and otherwise falls back to ink detection plus Knowledge Graph priors.

---

## Google Colab

Open [`Block_3_Handwriting_Recognition.ipynb`](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block3/block3/Block_3_Handwriting_Recognition.ipynb):

1. Run setup (pulls branch `block3`).
2. Upload `block2_validated_batch.zip` when the file popup appears.
3. Run batch HTR + prior fusion.
4. Download `block3_predictions_batch.zip`.

---

## Output ZIP contract (`block3_predictions_batch.zip`)

```text
manifest.json
docs/
  <doc_id>/
    prediction.json          # checkbox_marks, handwriting_fields, tubes, HiTL list
    annotated_canvas.png     # green = high-confidence, amber = needs review
    canonical.png
    metadata.json
    validation_report.json
    prior_rankings.json
    crops/checkboxes/
    crops/handwriting/
```

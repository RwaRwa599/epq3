# Block 3 — Nonverbal Marks & Verbal Handwriting

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block3/block3/Block_3_Handwriting_Recognition.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Block 3 reads **checkbox marks** (nonverbal) and **handwriting drafts** (verbal, TrOCR) from a saved Block 1 ZIP. It does **not** run Block 2 `assume()` or tube priors — that is Block 4, using these drafts plus the frozen KG JSON.

**Do not upload real clinic PHI to Colab.** Use the committed blank / synthetic sample.

---

## Two independent paths

```
block1_normalized_batch.zip          block2/kg/*.json
        |                                    |
        +------------------+-----------------+
                           v
                    +-------------+
                    |   BLOCK 3   |
                    | nonverbal   |  interior slash / filled (density fallback)
                    | verbal      |  TrOCR drafts (tubes = digits; others raw)
                    +-------------+
                           v
              hypotheses.json  →  Block 4 KG assume / tubes / rescoring
```

Qwen is out of this pass. Block 4 (KG constraints / rescoring) is later.

---

## Quick start

```bash
cd block3
pip install -r requirements.txt
python run_demo.py --input-zip block1_normalized_batch.zip --output-zip block3_predictions_batch.zip
pytest -v
```

Optional extras:

```bash
pip install paddlepaddle "paddleocr>=2.7,<3"
pip install transformers torch
```

Without those extras the density fallback and empty/ink verbal path still run (CI).

---

## Google Colab

Open [`Block_3_Handwriting_Recognition.ipynb`](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block3/block3/Block_3_Handwriting_Recognition.ipynb):

1. Setup clones branch `block3` and puts `src/` on `sys.path`.
2. **Nonverbal only** — PaddleOCR, no TrOCR.
3. **Verbal only** — TrOCR, no Paddle.
4. **Together** — Block 1 sample ZIP + Block 2 KG → `hypotheses.json`.

---

## Output (`block3_predictions_batch.zip`)

```text
manifest.json
docs/<doc_id>/
  hypotheses.json       # nonverbal + verbal + confidences
  prediction.json       # Block 4-shaped document prediction
  annotated_canvas.png
  canonical.png
  metadata.json
  crops/checkboxes/
  crops/handwriting/
```

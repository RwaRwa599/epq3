# Block 1 — Document Normalization & Batch ROI Extraction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block2/block1/Block_1_Document_Normalization.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

Block 1 is the **document normalization and ROI extraction engine** for medical laboratory request sheets. It handles digital scans and unconstrained mobile camera photos (perspective tilt, non-uniform shadows, rotations).

---

## Batch Pipeline: Block 1 -> Block 2 -> Block 3

```
[ Raw Photos / Scans (ZIP or Folder) ]
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ BLOCK 1: Normalization & Batch Crop Extraction         │
│ • Detects document quadrilateral & homography warp     │
│ • Corrects 0° / 180° rotation                         │
│ • Fine-aligns landmarks & snaps checkbox gutters       │
│ • Normalizes illumination (CLAHE + background divide)  │
│ • Exports standardized 'block1_normalized_batch.zip'   │
└────────────────────────────────────────────────────────┘
                 │
                 ▼ (block1_normalized_batch.zip)
┌────────────────────────────────────────────────────────┐
│ BLOCK 2: Clinical Knowledge Graph & Prior Engine       │
│ • Upload popup receives block1_normalized_batch.zip    │
│ • Expands profiles, calculates tubes, validates rules  │
│ • Exports 'block2_validated_batch.zip' for Block 3     │
└────────────────────────────────────────────────────────┘
```

---

## Standalone Usage

### 1. Batch Normalization CLI
```bash
python run_demo.py path/to/images/ --output-zip block1_normalized_batch.zip
```

### 2. Google Colab
Open [`Block_1_Document_Normalization.ipynb`](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block2/block1/Block_1_Document_Normalization.ipynb) in Colab:
1. Run setup cells.
2. Choose your input images via the upload popup.
3. Click to run batch normalization.
4. Download `block1_normalized_batch.zip`.

---

## Output ZIP Contract (`block1_normalized_batch.zip`)

```text
manifest.json
docs/
  <doc_id>/
    canonical.png        # 2048×1754 rectified RGB image
    overlay.png          # Visual verification overlay with bboxes
    metadata.json        # Quad corners, alignment score, detected candidate marks
    crops/
      checkboxes/
        <field_id>.png   # Normalized checkbox crops (138 fields)
      handwriting/
        <field_id>.png   # Normalized handwriting crops (12 fields)
```

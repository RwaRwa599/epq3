# Block 2 — Clinical Knowledge Graph & Prior Validation Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block2/block2/Block_2_Medical_Knowledge_Graph.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

Block 2 serves as the **frozen medical ground truth and clinical prior constraint engine**. It seamlessly ingests `block1_normalized_batch.zip` from Block 1, expands profile bundles, computes specimen tube rules, applies Bayesian prior rankings (`assume()`), runs cross-field clinical validation, and packages `block2_validated_batch.zip` for **Block 3 (Handwriting Recognition & HTR)**.

---

## Batch Pipeline: Block 1 -> Block 2 -> Block 3

```
[ block1_normalized_batch.zip from Block 1 ]
                     │
                     ▼ (Interactive Upload Popup)
┌────────────────────────────────────────────────────────┐
│ BLOCK 2: Batch Knowledge Graph Engine                  │
│ • Ingests Block 1 manifest, metadata, and crops        │
│ • Profile Expansion (Lipid, Liver, Renal, Anemia, etc) │
│ • Specimen Tube Computation (EDTA, CB, Fl, Cit, etc)   │
│ • Prior Ranking (assume()) for handwriting write-ins   │
│ • Cross-Field Clinical Validation & Discrepancies     │
└────────────────────────────────────────────────────────┘
                     │
                     ▼
[ block2_validated_batch.zip for Block 3 (HTR) ]
```

---

## Standalone Usage

### 1. Batch Validation CLI (Ingest Block 1 ZIP)
```bash
python run_demo.py --input-zip block1_normalized_batch.zip --output-zip block2_validated_batch.zip
```

### 2. Single-Sheet Simulation CLI
```bash
python run_demo.py \
  --tests cbc profile_lipid glucose_fasting \
  --tubes EDTA:1 CB:1 Fl:1 \
  --resolve SGPT \
  --fuzzy "cultur and sensitivty"
```

### 3. Google Colab
Open [`Block_2_Medical_Knowledge_Graph.ipynb`](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block2/block2/Block_2_Medical_Knowledge_Graph.ipynb) in Colab:
1. Run setup cells.
2. An interactive file popup asks you to upload `block1_normalized_batch.zip`.
3. The engine automatically processes the entire batch, running validation and priors.
4. Download `block2_validated_batch.zip` ready for Block 3.

---

## Output ZIP Contract (`block2_validated_batch.zip` for Block 3)

```text
manifest.json
docs/
  <doc_id>/
    canonical.png             # Rectified document image
    metadata.json             # Document metadata and field coordinates
    validation_report.json    # Complete clinical validation JSON report
    prior_rankings.json       # Bayesian prior rankings for handwriting fields
    crops/
      handwriting/
        <field_id>.png        # Handwriting crops for Block 3 HTR
```

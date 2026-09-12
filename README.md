# Medical Document Intelligence System (`new2`)

A modular, agent-assisted medical document intelligence system for laboratory request forms.

## Architecture

- **Block 1 — Document Normalization & ROI Extraction:** 1a warp/align + piecewise RANSAC, 1b section squares + extra-ink, 1c adaptive crop-window gate with neighbour prior. Does not classify ticks. Version history: [`docs/blocks/block1-versions.md`](docs/blocks/block1-versions.md).
- **Block 2 — Clinical Knowledge Graph:** Frozen `kg/lab_request_v1_kg.json`. Status: [`docs/blocks/block2-status.md`](docs/blocks/block2-status.md).
- **Block 3 — Nonverbal marks & verbal handwriting:** Residual vs blank template, logistic geometry features, optional Paddle fusion. Ingests a Block 1 ZIP. Version history: [`docs/blocks/block3-versions.md`](docs/blocks/block3-versions.md). Current synthetic scorecard: [`docs/blocks/data/synthetic-10tick-scorecard.json`](docs/blocks/data/synthetic-10tick-scorecard.json).

Hub: [`docs/blocks/README.md`](docs/blocks/README.md).

Do not add agent-memory files (`memory/`, `agent-skills/`, `AGENTS.md`, protocol docs, hooks, or agent-memory CI) to this repository or to new block packages.

---

## Standalone Colab Packages

- **`block1/`**: Standalone Block 1 package with CLI demo and `Block_1_Document_Normalization.ipynb`
- **`block2/`**: Standalone Block 2 package with CLI demo and `Block_2_Medical_Knowledge_Graph.ipynb`
- **`block3/`**: Standalone Block 3 package with CLI demo and `Block_3_Handwriting_Recognition.ipynb`

---

## Usage

### Block 1: Normalize Document & Extract Crops
```python
from med_doc import normalize_document

result = normalize_document("sheet.jpg")
tick = result.checkbox_crops["body_check_plan_1"]
print(tick.canonical_bbox, tick.quality_score)
```

### Block 2: Clinical Knowledge Graph & Prior Engine
```python
from med_doc.kg import KnowledgeGraph

kg = KnowledgeGraph.load()

# Expand profile bundle
lipid_tests = kg.expand_profile("profile_lipid")

# Compute required specimen tubes
tubes = kg.calculate_expected_tubes(["cbc", "alt", "glucose_fasting"])

# Cross-field validation
report = kg.validate_request(
    ticked_ids=["cbc", "alt", "glucose_fasting"],
    observed_tubes={"EDTA": 1, "CB": 1, "Fl": 1}
)
print(f"Valid: {report.is_valid}, Confidence: {report.confidence}")
```

### Block 3: Nonverbal marks + verbal HTR
```python
from med_doc.htr import process_from_block1
from med_doc.kg import KnowledgeGraph

kg = KnowledgeGraph.load()  # Block 2 JSON import, not a per-sheet runner
result = process_from_block1(
    "block1_normalized_batch.zip",
    output_zip="block3_predictions_batch.zip",
    kg=kg,
    mode="both",  # or "nonverbal" / "verbal"
)
print(result["manifest"]["total_documents"], result["output_zip"])
```

---

## Local Verification

```bash
pip install -e ".[dev]"
pytest -v
```

Private / unredacted clinic photos belong in `data/samples/private/` (gitignored). Never commit PHI.

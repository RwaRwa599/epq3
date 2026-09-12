# Medical Document Intelligence System (`new2`)

A modular, agent-assisted medical document intelligence system for laboratory request forms.

## Architecture

- **Block 1 — Document Normalization & ROI Extraction:** Detects document quadrilateral, homography-warps to canonical canvas (`2048×1754` or `2048×1720`), fine-aligns landmarks and checkbox gutters, extracts normalized ROI crops for all checkboxes and handwriting regions.
- **Block 2 — Clinical Knowledge Graph & Prior Validation Engine:** Frozen medical knowledge base covering 138+ tests and profiles, specimen tube rules, alias/acronym normalization, messy handwriting fuzzy matching, Bayesian prior ranking (`assume()`), and cross-field clinical validation.
- **Block 3 — Handwriting Recognition & Prior Fusion:** Checkbox mark classification, digit/date/optional-TrOCR handwriting reads, fusion against Block 2 priors, HiTL triage, and `block3_predictions_batch.zip` for Block 4 / LIS.


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

### Block 3: Marks, HTR, and Prior Fusion
```python
from med_doc.htr import process_batch_from_block2

result = process_batch_from_block2(
    "block2_validated_batch.zip",
    output_zip="block3_predictions_batch.zip",
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

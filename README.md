# Medical Document Intelligence System (`new2`)

A modular, agent-assisted medical document intelligence system for laboratory request forms.

## Architecture

- **Block 1 — Document Normalization & ROI Extraction:** Detects document quadrilateral, homography-warps to canonical canvas (`2048×1754` or `2048×1720`), fine-aligns landmarks and checkbox gutters, extracts normalized ROI crops for all checkboxes and handwriting regions.
- **Block 2 — Clinical Knowledge Graph & Prior Validation Engine:** Frozen medical knowledge base covering 138+ tests and profiles, specimen tube rules (`EDTA`, `CB`, `Fl`, `Cit`, `Urine`, `Stool`, `pap`, `UBT`), alias/acronym normalization, messy handwriting fuzzy matching, Bayesian prior ranking (`assume()`), and cross-field clinical validation.

---

## Standalone Colab Packages

- **`block1/`**: Standalone Block 1 package with CLI demo and `Block_1_Document_Normalization.ipynb`
- **`block2/`**: Standalone Block 2 package with CLI demo and `Block_2_Medical_Knowledge_Graph.ipynb`

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

---

## Local Verification

```bash
pip install -e ".[dev]"
pytest -v
```

Private / unredacted clinic photos belong in `data/samples/private/` (gitignored). Never commit PHI.

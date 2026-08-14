# Block 2 — Clinical Knowledge Graph & Prior Validation Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block2/block2/Block_2_Medical_Knowledge_Graph.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

Block 2 serves as the **frozen medical ground truth and clinical prior constraint engine** for the medical document intelligence system. It provides deterministic, zero-hallucination medical lookups, profile expansions, specimen tube rules, fuzzy handwriting resolution, and cross-field clinical validation.

---

## Key Features

1. **Frozen Clinical Knowledge Base:** Pre-compiled and verified JSON catalog covering all **138 printed test checkboxes**, composite health check profiles, and unconstrained write-ins.
2. **Profile Bundle Expansion:** Deterministically expands multi-test profiles (`Lipid Profile`, `Liver Function Test`, `Diabetes Profile`, `Anemia Profile`, `Renal Profile`) to their individual component test IDs.
3. **Specimen Tube Computation:** Automatically computes required laboratory specimen containers (`EDTA`, `CB`, `Fl`, `Cit`, `Urine`, `Stool`, `pap`, `UBT`) based on ordered test combinations.
4. **Acronym & Alias Resolution:** Normalizes doctor shorthand and abbreviations (`SGPT` → `alt`, `HbA1c` → `hba1c`, `VDRL` → `rpr`, `TSH` → `tsh`).
5. **Fuzzy Handwriting Write-in Matching:** Resolves messy, misspelled doctor handwriting in unconstrained *OTHERS* fields against known clinical tests.
6. **Contextual Prior Ranking (`assume()`):** Uses Bayesian priors to rank and boost handwriting hypotheses based on already-checked tests.
7. **Cross-Field Clinical Validation:** Flags tube shortages, redundant test orders, and missing required containers, generating structured audit reports.

---

## Folder Layout

```text
block2/
├── Block_2_Medical_Knowledge_Graph.ipynb # Google Colab Interactive Notebook
├── README.md                             # This documentation
├── requirements.txt                      # Dependencies (pydantic, matplotlib, pytest)
├── pyproject.toml                        # Standalone package definition
├── run_demo.py                           # Standalone CLI interactive demo
├── kg/
│   ├── lab_request_v1_kg.json            # Verified v1 Knowledge Graph (138 tests + profiles)
│   └── lab_request_v0_kg.json            # Verified v0 Knowledge Graph
├── med_doc/
│   ├── __init__.py                       # Package exports
│   ├── paths.py                          # Self-contained path resolver
│   └── kg/
│       ├── __init__.py                   # Subpackage exports
│       ├── graph.py                      # Core KnowledgeGraph engine
│       ├── schemas.py                    # Pydantic data schemas
│       └── textutil.py                   # Normalization & similarity matching
└── tests/
    └── test_kg.py                        # Standalone pytest suite
```

---

## Quick Start (Local)

### 1. Installation
```bash
cd block2
pip install -r requirements.txt
pip install -e .
```

### 2. Run Interactive CLI Demo
```bash
python run_demo.py \
  --tests cbc profile_lipid glucose_fasting \
  --tubes EDTA:1 CB:1 Fl:1 \
  --resolve SGPT \
  --fuzzy "cultur and sensitivty"
```

### 3. Run Unit Tests
```bash
pytest -v
```

---

## Google Colab Usage

You can run the notebook directly in Google Colab with one click:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block2/block2/Block_2_Medical_Knowledge_Graph.ipynb)

Inside Colab, the notebook automatically:
1. Installs minimal dependencies (`pydantic`).
2. Loads the frozen knowledge base.
3. Tests profile expansion, tube calculation, handwriting resolution, and prior ranking.
4. Validates simulated test sheets and exports structured JSON reports.

---

## Python API Usage

```python
from med_doc.kg import KnowledgeGraph

# 1. Load Knowledge Graph
kg = KnowledgeGraph.load()

# 2. Expand a health profile into component tests
lipid_tests = kg.expand_profile("profile_lipid")
# Output: ['chol_total', 'hdl', 'ldl', 'triglycerides']

# 3. Calculate required specimen tubes
tubes = kg.calculate_expected_tubes(["cbc", "alt", "glucose_fasting"])
# Output: {'EDTA': 1, 'CB': 1, 'Fl': 1}

# 4. Resolve clinical alias
canonical_id = kg.resolve_alias("SGPT")
# Output: 'alt'

# 5. Fuzzy match messy handwriting
matches = kg.fuzzy_match_catalogue("cultur and sensitivity", top_k=2)
# Output: [RankedCandidate(value='Culture And Sensitivity', score=0.955, tier=1)]

# 6. Validate full request against physical tube counts
report = kg.validate_request(
    ticked_ids=["cbc", "alt", "glucose_fasting"],
    observed_tubes={"EDTA": 1, "CB": 1, "Fl": 1}
)
print(f"Valid: {report.is_valid}, Confidence: {report.confidence}")
```

---

## Data Schema Contracts

All models use **Pydantic v2**:

- **`CatalogueItem`**: Definition of a test or profile (`field_id`, `label`, `section`, `kind`, `aliases`, `components`, `tubes`).
- **`RankedCandidate`**: Scored hypothesis (`value`, `canonical_id`, `score`, `tier`, `reason`).
- **`ValidationResult`**: Comprehensive audit report (`is_valid`, `ticked_tests`, `implied_tests`, `all_ordered_tests`, `expected_tubes`, `observed_tubes`, `warnings`, `discrepancies`, `confidence`).

---

## Integration with Block 1 & Downstream Blocks

- **Block 1 (Normalization & Crops):** Crops checkbox and handwriting regions mapped directly to template `field_id`s.
- **Block 2 (Knowledge Graph):** Maps `field_id`s to medical names, profile components, tube rules, and provides prior ranking for downstream HTR.
- **Block 3 / Downstream:** Receives candidate rankings from `kg.assume()` and validates whole-sheet consistency with `kg.validate_request()`.

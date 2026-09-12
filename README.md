# Medical Document Intelligence System (`new2`)

A modular, agent-assisted medical document intelligence system for laboratory request forms.

## Architecture

- **Block 1 — Document Normalization & ROI Extraction:** 1a warp/align + piecewise RANSAC, 1b section squares + extra-ink, 1c adaptive crop-window gate with neighbour prior. Does not classify ticks. Version history: [`docs/blocks/block1-versions.md`](docs/blocks/block1-versions.md).
- **Block 2 — Clinical Knowledge Graph:** Frozen `kg/lab_request_v1_kg.json`. Status: [`docs/blocks/block2-status.md`](docs/blocks/block2-status.md).
- **Block 3 — Nonverbal marks & verbal handwriting:** Residual ticks; tubes = digits, dates = grammar, `others` = charset + visual lexicon (KG match is Block 4). Version history: [`docs/blocks/block3-versions.md`](docs/blocks/block3-versions.md). Marks: [`docs/blocks/data/synthetic-10tick-scorecard.json`](docs/blocks/data/synthetic-10tick-scorecard.json). Verbal: [`docs/blocks/data/verbal-accuracy.json`](docs/blocks/data/verbal-accuracy.json).
- **Block 4 — KG rescoring:** Trusted ticks + write-in `assume()` + observed tubes vs expected. Does not invent counts from empty crops or HiTL ticks. Status: [`docs/blocks/block4-status.md`](docs/blocks/block4-status.md).
- **Block 5 — Review and LIS commit:** HiTL queue, human/nurse patches, optional LLM n-best rank (never authority). Status: [`docs/blocks/block5-status.md`](docs/blocks/block5-status.md).

Hub: [`docs/blocks/README.md`](docs/blocks/README.md).

Do not add agent-memory files (`memory/`, `agent-skills/`, `AGENTS.md`, protocol docs, hooks, or agent-memory CI) to this repository or to new block packages.

---

## Google Colab (live tree)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/Pipeline_Blocks_1_to_5.ipynb)

Colab does **not** download `src/` when you open a notebook from GitHub. Use branch **`block1`**:

| Notebook | Colab |
|---|---|
| Blocks 1–5 pipeline | [Open](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/Pipeline_Blocks_1_to_5.ipynb) |
| Block 1 | [Open](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_1_Document_Normalization.ipynb) |
| Block 2 | [Open](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_2_Knowledge_Graph.ipynb) |
| Block 3 | [Open](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_3_Marks_and_HTR.ipynb) |
| Block 4 | [Open](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_4_KG_Rescoring.ipynb) |
| Block 5 | [Open](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_5_Review_and_LIS.ipynb) |

Index: [`notebooks/README.md`](notebooks/README.md). First cell prints **`BOOTSTRAP_V3`** and clones branch `block1` onto `sys.path` (Colab does not ship `src/` with the notebook). If `No module named med_doc` persists: **Runtime → Disconnect and delete runtime**, reopen from GitHub `block1`, **Run all**.

```python
%cd /content/epq3
!git fetch origin block1 && git checkout block1 && git pull --ff-only origin block1
```

Private repo: Colab secret `GITHUB_TOKEN`. Do not upload clinic PHI.

The `block1/`, `block2/`, `block3/` trees are older standalone snapshots (their Open-in-Colab badges point at those snapshot branches, not this live tree).

---

## Usage

### Blocks 1–5: Batch photos → LIS orders
```python
from med_doc import run_blocks_1_to_5

result = run_blocks_1_to_5(
    "photos/",  # or "photos.zip", or ["a.jpg", "b.png"]
    output_dir="pipeline_out",
)
print(result["block5"]["manifest"]["total_documents"], result["zips"]["block5"])
```

Warp-only (no crops / HTR): `run_block1a_batch` with the same input shapes.

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

### Block 4: Rescore drafts with the frozen KG
```python
from med_doc import process_from_block3
from med_doc.kg import KnowledgeGraph

result = process_from_block3(
    "block3_predictions_batch.zip",
    output_zip="block4_predictions_batch.zip",
    kg=KnowledgeGraph.load(),
)
print(result["manifest"]["documents"][0]["expected_tubes"])
```

### Block 5: Review queue + LIS order
```python
from med_doc import process_from_block4
from med_doc.review import ReviewPatch

result = process_from_block4(
    "block4_predictions_batch.zip",
    output_zip="block5_orders_batch.zip",
    reviews={"sample_sheet_01": [ReviewPatch(field_id="tube_edta", action="set_tube", value="1")]},
)
print(result["manifest"]["documents"][0]["needs_review"])
```

---

## Local Verification

```bash
pip install -e ".[dev]"
pytest -v
```

Private / unredacted clinic photos belong in `data/samples/private/` (gitignored). Never commit PHI.

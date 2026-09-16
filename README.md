# Medical Document Intelligence System (`new2`)

A modular, agent-assisted medical document intelligence system for laboratory request forms.

## Architecture

- **Block 1 — Document Normalization & ROI Extraction:** 1a warp/align + piecewise RANSAC, 1b section squares + extra-ink, 1c adaptive crop-window gate with neighbour prior. Does not classify ticks. Version history: [`docs/blocks/block1-versions.md`](docs/blocks/block1-versions.md).
- **Block 2 — Clinical Knowledge Graph:** Frozen `kg/lab_request_v1_kg.json`. Status: [`docs/blocks/block2-status.md`](docs/blocks/block2-status.md).
- **Block 3 — Nonverbal marks & verbal handwriting:** Residual ticks; tubes = digits, dates = grammar, `others` = charset + visual lexicon (KG match is Block 4). Version history: [`docs/blocks/block3-versions.md`](docs/blocks/block3-versions.md). Marks: [`docs/blocks/data/synthetic-10tick-scorecard.json`](docs/blocks/data/synthetic-10tick-scorecard.json). Verbal: [`docs/blocks/data/verbal-accuracy.json`](docs/blocks/data/verbal-accuracy.json).
- **Block 4 — KG rescoring:** Trusted ticks + write-in `assume()` + observed tubes vs expected. Does not invent counts from empty crops or HiTL ticks. Status: [`docs/blocks/block4-status.md`](docs/blocks/block4-status.md).
- **Block 5 — Review and LIS commit:** HiTL queue, human/nurse patches, optional LLM n-best rank (never authority). Status: [`docs/blocks/block5-status.md`](docs/blocks/block5-status.md).

Hub: [`docs/blocks/README.md`](docs/blocks/README.md). **Version history, Colab, per-version accuracy, and why each failed:** [`CHANGELOG.md`](CHANGELOG.md) and [`docs/blocks/clinic-eval-chain.md`](docs/blocks/clinic-eval-chain.md).

Do not add agent-memory files (`memory/`, `agent-skills/`, `AGENTS.md`, protocol docs, hooks, or agent-memory CI) to this repository or to new block packages.

---

## Google Colab (live tree)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/Run_in_Colab.ipynb)

**Tick-only / combination critic (new):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/Tick_Pipeline_Colab.ipynb) — `htr_mode=nonverbal`, Block 4 `combo_backend=kg` (or ollama). First cell must print **`BOOTSTRAP_V7`**.

Colab does **not** download `src/` when you open a notebook from GitHub. Open **`Run_in_Colab.ipynb`** on branch **`block1`**. **Runtime → Disconnect and delete runtime**, then **Run all**. The first cell must print **`BOOTSTRAP_V6`** and **`tick_policy: slash-v2`**. Cell 8 sets **`OUTPUT_MODE = "dev"`** so you get `block1.zip` / `block3.zip` with `overlay.png` and `annotated_canvas.png` (user mode is only `order.json`). If `order.json` has no `tick_policy` field, Colab is still on stale code.

| Notebook | Colab |
|---|---|
| Tick pipeline (3a + combo critic) | [Open](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/Tick_Pipeline_Colab.ipynb) |
| Blocks 1–5 (use this) | [Open](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/Run_in_Colab.ipynb) |
| Same pipeline (older name) | [Open](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/Pipeline_Blocks_1_to_5.ipynb) |
| Block 1 | [Open](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_1_Document_Normalization.ipynb) |
| Block 2 | [Open](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_2_Knowledge_Graph.ipynb) |
| Block 3 | [Open](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_3_Marks_and_HTR.ipynb) |
| Block 4 | [Open](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_4_KG_Rescoring.ipynb) |
| Block 5 | [Open](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_5_Review_and_LIS.ipynb) |
| Layout crop (PHI strip) | [Open](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Layout_Crop_Batch.ipynb) |

Index: [`notebooks/README.md`](notebooks/README.md). Do not upload clinic PHI. The repo is public; no GitHub token is required.

The `block1/`, `block2/`, `block3/` trees are older standalone snapshots (their Open-in-Colab badges point at those snapshot branches, not this live tree).

---

## Usage

### Crop header/footer off a batch (before Blocks 1–5)

The repo has a **`block1/` folder** (old Colab snapshot) and a **`block1` branch**. `git checkout block1` is ambiguous; use `git switch`:

```bash
cd /Users/renaw/epq3
git fetch origin
git switch -C block1 origin/block1
python3 --version   # need 3.10+
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m med_doc.privacy /Users/renaw/Downloads/labform_pages \
  --out /Users/renaw/Downloads/labform_cropped --debug
```

Do not use `/Applications/Xcode.app/.../python3` for this — that pip is too old and has no `med_doc`. If `python3 --version` is 3.9, install Homebrew Python 3.12 and use that `python3`.

**No package install:** download one script. Default cut is the **top 10.5/29.5 of the page height** (ratio, same on every photo size). It then checks that “Clinical Information” and a column header are still in the PNG and pulls the cut up if not.

```bash
curl -L -o /tmp/crop_labform_pages.py \
  https://raw.githubusercontent.com/RwaRwa599/epq3/block1/scripts/crop_labform_pages.py
python3 -m pip install pillow numpy opencv-python-headless
python3 /tmp/crop_labform_pages.py /Users/renaw/Downloads/labform_pages \
  --out /Users/renaw/Downloads/labform_cropped --debug
```

Config: [`configs/layout_crop.json`](configs/layout_crop.json) (template band) or [`configs/layout_crop.layoutparser.json`](configs/layout_crop.layoutparser.json) (`keep_types` / `drop_types`). How-to: [`docs/blocks/layout-crop.md`](docs/blocks/layout-crop.md).

```bash
python -m med_doc.privacy photos/ --out cropped/
python -m med_doc.privacy photos/ --out cropped/ \
  --config configs/layout_crop.layoutparser.json --debug
```

```python
from med_doc.privacy import crop_batch, load_crop_config

cfg = load_crop_config()
cfg.template.top, cfg.template.bottom = 0.08, 1.0  # drop name header only; keep footer
crop_batch("photos/", "cropped/", config=cfg)
```

### Blocks 1–5: Batch photos → LIS orders
```python
from med_doc import run_blocks_1_to_5

result = run_blocks_1_to_5(
    "photos/",  # or "photos.zip", or ["a.jpg", "b.png"]
    output_dir="pipeline_out",  # default output_mode="user" → one order.json
)
print(result["output_json"])  # pipeline_out/order.json
# Developer dump (crops + per-block ZIPs):
# run_blocks_1_to_5("photos/", output_dir="pipeline_out", output_mode="dev")
```

Ticks are geometry + logistic (not TrOCR/Paddle). Crop gold + refit: [`data/labels/README.md`](data/labels/README.md). `python -m med_doc.eval models` prints the live stack.

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

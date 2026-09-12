# Blocks 1–5 — live tree, versions, data

This folder is the **GitHub-facing record** of Blocks 1–5: what shipped in each architecture generation, what the live `src/med_doc` tree does now, and the latest **non-PHI** eval numbers.

| Doc | Contents |
|---|---|
| [block1-versions.md](./block1-versions.md) | Every Block 1 architecture generation + current data |
| [block3-versions.md](./block3-versions.md) | Every Block 3 architecture generation + current data |
| [block2-status.md](./block2-status.md) | Frozen Block 2 KG (no separate version train in this pass) |
| [block4-status.md](./block4-status.md) | KG rescoring of Block 3 drafts (trusted ticks, tubes, write-ins) |
| [block5-status.md](./block5-status.md) | HiTL review, nurse/LLM-assist patches, LIS `order.json` |
| [data/synthetic-10tick-scorecard.json](./data/synthetic-10tick-scorecard.json) | Latest 1c + Block 3 mark scorecard |
| [data/verbal-accuracy.json](./data/verbal-accuracy.json) | Block 3 verbal (tubes / dates / others) |

**Live implementation** is `src/med_doc` on git branch **`block1`**. Public API:

- Block 1: `normalize_document()` / `normalize_batch()` → ZIP
- Block 2: `KnowledgeGraph.load()` (`kg/lab_request_v1_kg.json`)
- Block 3: `process_from_block1()` on that ZIP → `hypotheses.json`
- Block 4: `process_from_block3()` on that ZIP + frozen KG → `prediction.json`
- Block 5: `process_from_block4()` → `review.json` + `order.json`

Standalone Colab trees `block1/`, `block2/`, `block3/` are **older snapshots** for notebooks. They do not contain Block 1c or interior V/slash geometry. Prefer `src/med_doc` via [`Pipeline_Blocks_1_to_5.ipynb`](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/Pipeline_Blocks_1_to_5.ipynb) (clone branch `block1`; Colab does not fetch `src/` with the notebook alone).

Clinic photos stay in gitignored `data/samples/private/`. They are not part of this upload.

```mermaid
flowchart LR
  photo[Photo or blank PNG]
  b1[Block 1 normalize ZIP]
  kg[Block 2 frozen KG JSON]
  b3[Block 3 hypotheses ZIP]
  b4[Block 4 prediction.json]
  b5[Block 5 order.json]
  photo --> b1 --> b3 --> b4 --> b5
  kg --> b4
  kg --> b5
```

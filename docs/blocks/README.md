# Blocks 1–3 — live tree, versions, data

This folder is the **GitHub-facing record** of Blocks 1–3: what shipped in each architecture generation, what the live `src/med_doc` tree does now, and the latest **non-PHI** eval numbers.

| Doc | Contents |
|---|---|
| [block1-versions.md](./block1-versions.md) | Every Block 1 architecture generation + current data |
| [block3-versions.md](./block3-versions.md) | Every Block 3 architecture generation + current data |
| [block2-status.md](./block2-status.md) | Frozen Block 2 KG (no separate version train in this pass) |
| [data/synthetic-10tick-scorecard.json](./data/synthetic-10tick-scorecard.json) | Latest 1c + Block 3 mark scorecard |
| [data/verbal-accuracy.json](./data/verbal-accuracy.json) | Block 3 verbal (tubes / dates / others) |

**Live implementation** is `src/med_doc` on git branch **`block1`**. Public API:

- Block 1: `normalize_document()` / `normalize_batch()` → ZIP
- Block 2: `KnowledgeGraph.load()` (`kg/lab_request_v1_kg.json`)
- Block 3: `process_from_block1()` on that ZIP → `hypotheses.json`

Standalone Colab trees `block1/`, `block2/`, `block3/` are **older snapshots** for notebooks. They do not contain Block 1c or interior V/slash geometry. Prefer `src/med_doc`.

Clinic photos stay in gitignored `data/samples/private/`. They are not part of this upload.

```mermaid
flowchart LR
  photo[Photo or blank PNG]
  b1[Block 1 normalize ZIP]
  kg[Block 2 frozen KG JSON]
  b3[Block 3 hypotheses ZIP]
  photo --> b1 --> b3
  kg -.-> b3
```

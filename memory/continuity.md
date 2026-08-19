# Continuity — new2

> Shared ground truth for project state across all agents and sessions.
> Update at the end of every session. Never delete — only archive (see `REVIEW.md`).
>
> Each fact carries a metadata footer in an HTML comment, maintained by the review
> ritual — invisible when rendered, read/written by agents:
> `<!-- id: kebab-id | created: YYYY-MM-DD | last_used: YYYY-MM-DD | uses: N | tier: active -->`
> See `.agent/schema.md` for the fields and `memory/decay-policy.md` for the windows.

---

## Project State

- **project:** new2
- **status:** Block 3 verbal/nonverbal on Block 1 ZIP + Block 2 KG import; Colab notebook ready
- **last_enabled:** 2026-08-14
- **last_session:** 2026-08-19 | agent: Cursor (2026-08-19-110415)
- **last_review:** (none yet)
- **last_invariant_check:** (none yet)
- **repo:** ~/new2

## Stack & Tools

> Canonical live home for the current stack — language version, dependencies, tool
> versions. `instructions.md` keeps only a high-level descriptor and points here.

- Python >=3.10; package `med-doc` 0.1.0 (`pyproject.toml`)
- Runtime: numpy, opencv-python-headless, Pillow, pydantic; optional PaddleOCR (nonverbal) and TrOCR/transformers (verbal)
- Tests: pytest; Block 1+2+3 suite last recorded 40 passed
<!-- id: stack-python-opencv | created: 2026-08-14 | last_used: 2026-08-14 | uses: 4 | tier: active | origin: 2026-08-14-025804 -->

## Key Decisions

- Initialized with agent-memory v4.32.1 (Mode A, deep analysis)
  <!-- id: init-agent-memory-v4321 | created: 2026-08-14 | last_used: 2026-08-14 | uses: 3 | tier: active | origin: 2026-08-14-023004 -->
- Default Block 1 overlay is digital print **v0** (`templates/lab_request_canonical.json`, canvas 2048×1720). Clinic photos are a later **v1** print (`templates/lab_request_v1_canonical.json`).
  <!-- id: decision-v0-default-v1-clinic | created: 2026-08-14 | last_used: 2026-08-14 | uses: 4 | tier: active | origin: 2026-08-14-025804 -->
- Full-page scans use height letterbox to the canonical canvas, not an anisotropic stretch.
  <!-- id: decision-letterbox-fullpage | created: 2026-08-14 | last_used: 2026-08-14 | uses: 1 | tier: working | origin: 2026-08-14-025804 -->
- Clinic photos automatically select `v1` template via `_pick_revision` and use 1-to-1 checkbox snapping within column boundaries (`snap_overlay`) with contrast-relative scoring.
  <!-- id: decision-photo-v1-snap-matching | created: 2026-08-14 | last_used: 2026-08-14 | uses: 1 | tier: working | origin: 2026-08-14-031206 -->
- Block 2 Knowledge Graph is frozen and deterministic (`kg/lab_request_v1_kg.json`), providing profile expansions, tube requirements, acronym resolution, fuzzy matching for write-ins, and cross-field validation.
  <!-- id: block2-clinical-kg | created: 2026-08-14 | last_used: 2026-08-14 | uses: 3 | tier: active | origin: 2026-08-14-034450 -->
- Prior engine `assume()` biases downstream HTR hypotheses using Bayesian priors from observed checkboxes and profile bundles.
  <!-- id: decision-frozen-kg-priors | created: 2026-08-14 | last_used: 2026-08-14 | uses: 2 | tier: active | origin: 2026-08-14-034450 -->
- Standardized ZIP contract links Block 1 (`block1_normalized_batch.zip`) → Block 2 (`block2_validated_batch.zip`) → Block 3 (`block3_predictions_batch.zip` for Block 4 / LIS).
  <!-- id: block1-block2-batch-zip-pipeline | created: 2026-08-14 | last_used: 2026-08-14 | uses: 2 | tier: superseded | origin: 2026-08-14-035451 | superseded-by: block3-ingests-block1-zip-kg-import -->
- Block 3 classifies checkbox marks, reads handwriting crops (digits/dates/optional TrOCR), fuses drafts with Block 2 priors, and flags HiTL fields.
  <!-- id: block3-htr-prior-fusion | created: 2026-08-14 | last_used: 2026-08-14 | uses: 1 | tier: superseded | origin: 2026-08-14-080522 | superseded-by: block3-verbal-nonverbal-split -->
- Block 3 is two independent recognizers: nonverbal (PaddleOCR on checkbox crops, `source` `paddle` or `density_fallback`) and verbal (TrOCR on handwriting crops; tubes keep digits; `others` stays raw). Qwen is out of this pass.
  <!-- id: block3-verbal-nonverbal-split | created: 2026-08-19 | last_used: 2026-08-19 | uses: 1 | tier: working | origin: 2026-08-19-110415 | supersedes: block3-htr-prior-fusion -->
- Block 3 ingests a saved Block 1 ZIP (`block1_normalized_batch.zip`) and imports Block 2 as frozen KG JSON (`KnowledgeGraph.load()` / `assume()`), not Block 2's per-sheet batch. Emits `hypotheses.json`. Linear 1→3→4 with Block 2 as a file is a DAG; Colab ZIP relay through Block 2 is not required.
  <!-- id: block3-ingests-block1-zip-kg-import | created: 2026-08-19 | last_used: 2026-08-19 | uses: 1 | tier: working | origin: 2026-08-19-110415 | supersedes: block1-block2-batch-zip-pipeline -->

## Conventions

(none yet — to be established)

## Open Threads

> Mark completed items `- [x]` and leave them in place — the review sweeps them to
> the archive once older than `archive_window` sessions. Don't archive them by hand.

- [ ] (vision-bootstrap) Confirm the Vision in memory/vision.md — set the target / success criteria / non-goals; then derive the Blueprint.
  <!-- id: vision-bootstrap | created: 2026-08-14 | last_used: 2026-08-14 | uses: 2 | tier: active | origin: 2026-08-14-023004 -->
- [x] Greenfield — no code yet: record the stack in ## Stack & Tools, coding conventions, Architectural Invariants, and seed the stack's build-output .gitignore entries when the stack lands.
  <!-- id: greenfield-seed-stack | created: 2026-08-14 | last_used: 2026-08-14 | uses: 2 | tier: active | origin: 2026-08-14-023004 -->
- [x] Photographed clinic sheets use print v1 (extra rows: ANA, Molecular, Pap, extra tubes). Block 1 needs a v1 snap/warp before the v0 overlay will sit on those squares.
  <!-- id: clinic-print-v1-drift | created: 2026-08-14 | last_used: 2026-08-14 | uses: 2 | tier: active | origin: 2026-08-14-025804 -->
- [ ] Block 4: KG constraints / tubes / rescoring on Block 3 hypotheses (no VLM). Then Block 5 report + accept/loop_once/HiTL gate.
  <!-- id: block4-kg-constraints | created: 2026-08-19 | last_used: 2026-08-19 | uses: 1 | tier: working | origin: 2026-08-19-110415 -->

## User Preferences

- Colab is development only (synthetic / committed blank). Do not upload real clinic PHI to Colab. Clinic product is later local Jupyter.
  <!-- id: colab-no-phi | created: 2026-08-19 | last_used: 2026-08-19 | uses: 1 | tier: working | origin: 2026-08-19-110415 -->

## Team / Members

(none recorded yet)

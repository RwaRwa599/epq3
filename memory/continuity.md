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
- **status:** Block 1 & Block 2 batch ZIP pipeline implemented and verified (19/19 tests passed); Colab interactive popups and Block 3 export ZIP ready
- **last_enabled:** 2026-08-14
- **last_session:** 2026-08-14 | agent: Cursor (2026-08-14-035451)
- **last_review:** (none yet)
- **last_invariant_check:** (none yet)
- **repo:** ~/new2

## Stack & Tools

> Canonical live home for the current stack — language version, dependencies, tool
> versions. `instructions.md` keeps only a high-level descriptor and points here.

- Python >=3.10; package `med-doc` 0.1.0 (`pyproject.toml`)
- Runtime: numpy, opencv-python-headless, Pillow, pydantic
- Tests: pytest; Block 1 suite last recorded 11 passed
<!-- id: stack-python-opencv | created: 2026-08-14 | last_used: 2026-08-14 | uses: 3 | tier: active | origin: 2026-08-14-025804 -->

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
  <!-- id: block2-clinical-kg | created: 2026-08-14 | last_used: 2026-08-14 | uses: 2 | tier: active | origin: 2026-08-14-034450 -->
- Prior engine `assume()` biases downstream HTR hypotheses using Bayesian priors from observed checkboxes and profile bundles.
  <!-- id: decision-frozen-kg-priors | created: 2026-08-14 | last_used: 2026-08-14 | uses: 1 | tier: working | origin: 2026-08-14-034450 -->
- Standardized ZIP contract links Block 1 (`block1_normalized_batch.zip`) -> Block 2 (Colab upload popup, Knowledge Graph batch validation) -> Block 3 (`block2_validated_batch.zip` for HTR).
  <!-- id: block1-block2-batch-zip-pipeline | created: 2026-08-14 | last_used: 2026-08-14 | uses: 1 | tier: working | origin: 2026-08-14-035451 -->

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

## User Preferences

(none recorded yet — record ONLY what the user explicitly states; never infer)

## Team / Members

(none recorded yet)

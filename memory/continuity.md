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
- **status:** Block 1 is 1a/1b (no 1c on this branch); Block 3 density fallback has zero recall on synthetic ticks in 1b crops
- **last_enabled:** 2026-08-14
- **last_session:** 2026-08-31 | agent: Cursor (2026-08-31-153927)
- **last_review:** (none yet)
- **last_invariant_check:** (none yet)
- **repo:** ~/new2

## Stack & Tools

> Canonical live home for the current stack — language version, dependencies, tool
> versions. `instructions.md` keeps only a high-level descriptor and points here.

- Python >=3.10; package `med-doc` 0.1.0 (`pyproject.toml`)
- Runtime: numpy, opencv-python-headless, Pillow, pydantic; optional PaddleOCR (nonverbal) and TrOCR/transformers (verbal)
- Tests: pytest; Block 1+2+3 suite last recorded 43 passed
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
- Block 1 `snap_overlay` may use printed section bars (per column, order-paired, Y affine) only when `_checkbox_grid_score` beats hollow-ring snap by ≥ 0.02. Forced text-name / ungated bar snap lost hollow score; gated path kept synthetic 95.2% and clinic mean 54.3% → 55.2%.
  <!-- id: block1-section-bar-gated-snap | created: 2026-08-20 | last_used: 2026-08-20 | uses: 1 | tier: working | origin: 2026-08-20-031237 -->
- Template overlay has two-level printed territories: main = black header bar through the next bar (column-clipped); sub = bold subhead + its JSON groups. Built in `med_doc.normalization.sections` and drawn on `overlay.png` / `sections.png`. Extra-ink subtract later uses the digital blank (`lab_request_v0_blank.png`), not a filled clinic photo.
  <!-- id: template-section-territories | created: 2026-08-20 | last_used: 2026-08-20 | uses: 1 | tier: working | origin: 2026-08-20-032544 -->
- Block 1 `snap_sections` locks each main to a detected black header, cyan subs to bold subheads, and green rows to JSON tests in printed-peak order below that header (not snap_overlay field Y). Within each row the test-name text is boxed and subtracted; the leftover from the magenta column wall to that text is one tick box. Rows and ticks carry the JSON `field_id`.
  <!-- id: block1-header-to-next-section-snap | created: 2026-08-20 | last_used: 2026-08-20 | uses: 1 | tier: working | origin: 2026-08-20-035004 -->
- Block 1 `extract_crops` takes one ~16–24 px hollow square per checkbox `field_id` found inside that field’s red tick strip (`apply_tick_windows`). If `_checkbox_grid_score` drops more than 0.02 vs `snap_overlay`, crops stay on the snap_overlay bboxes. Overlay still draws `sectioned`; handwriting fields are not retargeted. Synthetic hollow ≥ 95%.
  <!-- id: block1-tick-window-gated-crops | created: 2026-08-20 | last_used: 2026-08-20 | uses: 1 | tier: superseded | origin: 2026-08-20-113324 | superseded-by: block1-1a-1b-split -->
- Block 1 internals are `run_block1a` (revision pick, warp, page `fine_align`) then `run_block1b` (section lock, per-section dy from printed-row peaks, tick-strip squares preferring ink-in-ring, extra-ink `crops/sections/{id}.png`). Public API stays `normalize_document()`. Square placement may fall back to dy-shifted snap fields when tick-window grid is >0.02 worse; extra-ink always keeps the section lock. Synthetic hollow ≥ 95%.
  <!-- id: block1-1a-1b-split | created: 2026-08-28 | last_used: 2026-08-28 | uses: 1 | tier: working | origin: 2026-08-28-144429 | supersedes: block1-tick-window-gated-crops -->
- Block 1 does not classify ticks. Fill/dark ratio in 1b is registration quality; `is_marked_candidate` is debug-only; ZIP `mark_classification` is `deferred_to_block3`.
  <!-- id: block1-no-mark-classification | created: 2026-08-28 | last_used: 2026-08-28 | uses: 1 | tier: working | origin: 2026-08-28-144429 -->
- Local demo9 through 1a/1b → Block 3: Paddle over-calls because `_looks_like_mark(t) or (t and s >= 0.55)` treats any OCR snippet as a tick when interior `ink_density >= 0.06`. Density correctly left those unmarked (`source=density`, printed label / mis-crop). Prior demo9_b1b3 had Paddle off (all `density_fallback`). 1b tick-windows unused; many crops shifted onto labels so density cleared 0.06 and Paddle read glyph fragments (`opc`, `-h(`). Block 1 `dark_ratio` / `is_marked_candidate` is not the product path when crops exist. Blank clean; s1 `urine_culture` still a density slash.
  <!-- id: demo9-1a1b-paddle-fps | created: 2026-08-28 | last_used: 2026-08-28 | uses: 1 | tier: working | origin: 2026-08-28-145408 -->
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
- Block 3 ingests a saved Block 1 ZIP (`block1_normalized_batch.zip`) and imports Block 2 as frozen KG JSON for Block 4, not Block 2's per-sheet batch. Emits `hypotheses.json`.
  <!-- id: block3-ingests-block1-zip-kg-import | created: 2026-08-19 | last_used: 2026-08-19 | uses: 1 | tier: working | origin: 2026-08-19-110415 | supersedes: block1-block2-batch-zip-pipeline -->
- Block 3 nonverbal is precision-first: a tick must be one interior slash (or a filled box) inside a printed square. Printed corners, label glyphs, and density-only ink are unmarked. Clinic 5-sheet dry run: 1/1 true tick, 0 false ticks.
  <!-- id: nonverbal-precision-first | created: 2026-08-19 | last_used: 2026-08-19 | uses: 1 | tier: working | origin: 2026-08-19-112047 -->
- 1a/1b → Block 3 on synthetic v0 (10 gold ticks, Paddle off): digital 0 TP / 10 FN / 0 FP; photo 0 TP / 10 FN / 1 FP (`ca72_4`). Interior ink is present on digital gold crops (~0.29 density) but rejected as `density`. Clinic gate5 not in this clone. Accuracy did not rise vs prior clinic dual-signal 10/10/41.
  <!-- id: block3-1ab-synth-recall-zero | created: 2026-08-31 | last_used: 2026-08-31 | uses: 1 | tier: working | origin: 2026-08-31-153927 -->
- Block 3 verbal emits OCR drafts only (`raw_text`, confidence, `unavailable`/`ink-present` → HiTL). Block 4 applies KG `assume()`, tube expected vs observed, and catalogue constraints. Do not fuse priors in Block 3.
  <!-- id: verbal-drafts-for-block4 | created: 2026-08-19 | last_used: 2026-08-19 | uses: 1 | tier: working | origin: 2026-08-19-112047 -->

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
- [x] Block 3 slash detector treats printed checkbox corners as ticks on clinic photos (tens of FPs/sheet). Density-only or Paddle interior-ink should be the default until slash is gated on interior pixels.
  <!-- id: clinic-slash-false-positives | created: 2026-08-19 | last_used: 2026-08-19 | uses: 1 | tier: working | origin: 2026-08-19-111033 -->
- [x] `has_ink` min_frac is too high for large `others` ROIs: handwriting is in the crop but verbal returns empty. Need an ink gate that is area-adaptive (or a tighter crop around dark pixels).
  <!-- id: others-has-ink-too-strict | created: 2026-08-19 | last_used: 2026-08-19 | uses: 1 | tier: working | origin: 2026-08-19-111033 -->
- [x] Block 3 already writes KG tube `prior_expected` from ticked IDs. False ticks therefore invent tube counts. Tube priors belong in Block 4 after marks are trusted.
  <!-- id: tube-priors-from-false-ticks | created: 2026-08-19 | last_used: 2026-08-19 | uses: 1 | tier: working | origin: 2026-08-19-111033 -->
- [ ] Block 4: KG constraints / tubes / rescoring on Block 3 hypotheses (no VLM). Then Block 5 report + accept/loop_once/HiTL gate.
  <!-- id: block4-kg-constraints | created: 2026-08-19 | last_used: 2026-08-19 | uses: 1 | tier: working | origin: 2026-08-19-110415 -->
- [ ] Block 1 remaining hollow misses on clinic photos are mostly full-frame warp / X, not missing section headers. Section-bar Y affine helps one sheet slightly; label-strip matching did not.
  <!-- id: block1-warp-limits-hollow | created: 2026-08-20 | last_used: 2026-08-20 | uses: 1 | tier: working | origin: 2026-08-20-031237 -->
- [ ] Header-to-next section snap is in; next is extra-ink vs the digital blank. Do not extra-ink until the HQ overlays look right to a human.
  <!-- id: block1-section-match-samples | created: 2026-08-20 | last_used: 2026-08-20 | uses: 1 | tier: working | origin: 2026-08-20-032544 -->
- [ ] Local (not CI): demo6 s6 / s7 first-field crop Y should sit on the first printed line under the header after tick-window gating, not one row down.
  <!-- id: block1-demo6-first-row-crop-y | created: 2026-08-20 | last_used: 2026-08-20 | uses: 1 | tier: working | origin: 2026-08-20-113324 -->
- [ ] Paddle over-calls ticks on 1a/1b clinic crops (demo9_1a1b: 7–43 ticks/sheet vs prior density-only). Need to check crop content vs Paddle gate, not density fallback.
  <!-- id: demo9-paddle-overcall | created: 2026-08-28 | last_used: 2026-08-28 | uses: 1 | tier: working | origin: 2026-08-28-145408 -->
- [ ] Block 3 density path misses V/lambda ticks in 1b overlay crops (border frac 0, slash gate false). Need a frame-aware slash/V accept without reopening Paddle FPs. Re-run on gate5 when PHI is present; 1c is not on this git snapshot.
  <!-- id: block3-recall-1b-crops | created: 2026-08-31 | last_used: 2026-08-31 | uses: 1 | tier: working | origin: 2026-08-31-153927 -->

## User Preferences

- Colab is development only (synthetic / committed blank). Do not upload real clinic PHI to Colab. Clinic product is later local Jupyter.
  <!-- id: colab-no-phi | created: 2026-08-19 | last_used: 2026-08-19 | uses: 1 | tier: working | origin: 2026-08-19-110415 -->

## Team / Members

(none recorded yet)

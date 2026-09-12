# Engineering brief — Block 1a / 1b / 1c contract

Block 1 **preprocesses only**. Public API stays `normalize_document()`; callers do not choose 1a vs 1b vs 1c.

Version train (architectures B1.1–B1.7): [`docs/blocks/block1-versions.md`](blocks/block1-versions.md).

## 1a — global + piecewise normalisation

`med_doc.normalization.block1a.run_block1a`:

1. Pick print revision (v0 vs v1)
2. Warp to the canonical canvas
3. Page-level `fine_align` (tx/ty; local paper / flattened column peaks)
4. Optional RANSAC partial-affine + piecewise residual warp when grid score improves

Returns a `Block1aPage` (`canvas`, `template`, align/warp meta, `col_shifts`). Column Y shifts stay on 1a (page-column, not section). 1b may override checkbox Y with per-section dy.

## 1b — layout

`med_doc.normalization.block1b.run_block1b`:

1. `snap_overlay` then `snap_sections` (header-bar lock)
2. Per-section projection **dy** on each leaf main/sub (tick-column ink vs expected row Y)
3. Split each row into tick strip vs label
4. One square crop per checkbox `field_id` inside that strip (prefer a printed ring with a dark interior over the first hollow; light border inset)
5. Per-section extra-ink crop: tick-column of that main/sub, slightly padded, **raw page pixels** (no blank-form subtraction)
6. Handwriting / `others` / tubes stay handwriting crops

Sheet-wide hollow-grid is **not** used to drop section lock. Square placement is per-field; if no hollow/ink square is found, the dy-shifted bbox is kept.

## 1c — crop-window gate

`med_doc.normalization.block1c.run_block1c` (after 1b):

1. Pass if the PNG is a printed square (empty hollow or ink-in-ring) and not a label strip — **locally adaptive paper**, not a page-global percentile
2. Else one widened rematch ranked by **RANSAC neighbour prior** plus distance to the 1b center (hollow-only if the field looks empty; neighbour-steal and dark-header guards)
3. Else keep the 1b bbox and set `crop_needs_hitl`

Does not classify ticks. Overlay-sized cells skip unless they look like a label.

## ZIP contract

| Path | Contents |
|---|---|
| `crops/checkboxes/{field_id}.png` | Square per JSON field |
| `crops/handwriting/{field_id}.png` | Text / tubes / others |
| `crops/sections/{section_id}.png` | Extra-ink tick-column; metadata lists member `field_ids` |

`NormalizedDocumentResult.section_crops` holds the in-memory extra-ink crops.

## What Block 1 does not do

- Classify ticks (`is_marked` / fill-ratio product signals). Interior dark ratio may appear under `debug` / `detected_marks` for older Block 3 ingest; it is registration quality, not a mark label. `mark_classification` is `deferred_to_block3`.
- Blank-form extra-ink subtraction
- Standalone `block1a/` / `block1b/` Colab trees (use `src/med_doc`; `block1/` notebook tree is a pre-1c snapshot)

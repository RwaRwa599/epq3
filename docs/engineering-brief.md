# Engineering brief — Block 1a / 1b contract

Block 1 **preprocesses only**. Public API stays `normalize_document()`; callers do not choose 1a vs 1b.

## 1a — global normalisation

`med_doc.normalization.block1a.run_block1a`:

1. Pick print revision (v0 vs v1)
2. Warp to the canonical canvas
3. Page-level `fine_align` (tx/ty; orientation is inside warp)

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
- Standalone `block1a/` / `block1b/` Colab trees

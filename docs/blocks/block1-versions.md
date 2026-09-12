# Block 1 — version history and architecture

**Current generation: B1.6** (1a warp → 1b layout → 1c crop gate). Public API has always been `normalize_document()`; callers never pick 1a/1b/1c.

Live code: `src/med_doc/normalization/` on branch `block1`. Contract notes: [`docs/engineering-brief.md`](../engineering-brief.md).

Print templates (not “Block 1 versions”): digital **v0** `templates/lab_request_canonical.json` (2048×1720, 124 checkboxes) vs clinic **v1** `templates/lab_request_v1_canonical.json` (2048×1754, 138 checkboxes).

---

## B1.1 — Warp + overlay snap (2026-08-14)

**Git:** `cd50e08` on `main` / early `block1`. Colab package `block1/`.

**Intent:** Photo → canonical canvas → one crop per template field.

```mermaid
flowchart LR
  img[RGB photo]
  quad[Detect page quad]
  warp[Homography to canvas]
  snap[snap_overlay hollow rings]
  crops[extract_crops]
  img --> quad --> warp --> snap --> crops
```

**Architecture**

- Document quad, `four_point_transform`, orientation.
- Overlay bboxes from JSON; hollow-ring snap in columns.
- Fine-align kept only when checkbox-grid score improves.
- Default overlay was later corrected to **v0**; clinic sheets need **v1**.

**Data then:** Digital blank crops were hollow squares (grid ~0.78). Photographed clinic print drifted on v0 (extra Molecular / Pap / tubes).

---

## B1.2 — v0/v1 pick + letterbox (2026-08-14)

**Facts:** `decision-v0-default-v1-clinic`, `decision-letterbox-fullpage`, `decision-photo-v1-snap-matching`.

**Architecture change:** `_pick_revision` / later `pick_revision` scores v0 vs v1 snap; full-page scans **letterbox by height** (no anisotropic stretch). Clinic photos use v1 1-to-1 checkbox snap in columns.

```mermaid
flowchart LR
  img[Photo]
  pick[pick_revision v0 vs v1]
  letterbox[Height letterbox]
  warp[Warp]
  snap[v1 column snap]
  img --> pick --> letterbox --> warp --> snap
```

---

## B1.3 — Gated section-bar snap (2026-08-20)

**Fact:** `block1-section-bar-gated-snap`. Report: `block1/section_snap_benchmark.md`.

**Architecture:** After ring snap, try per-column printed header bars → Y affine. **Keep only if** `_checkbox_grid_score` beats rings by ≥ 0.02.

**Data then**

| Setup | Hollow / grid |
|---|---|
| Synthetic rendered + photographed v0 | **95.2%** (unchanged by the gate) |
| Clinic 5-sheet mean | 54.3% → **55.2%** |

Test-name / letter-initial matching was prototyped and **not** shipped.

---

## B1.4 — Territories, header lock, tick windows (2026-08-20)

**Facts:** `template-section-territories`, `block1-header-to-next-section-snap`, superseded `block1-tick-window-gated-crops`.

**Architecture**

- Mains = black header through next bar; subs = bold subheads + JSON groups.
- `snap_sections`: lock mains to headers, rows to printed-peak order (not overlay field Y).
- Tick box = leftover from column wall to boxed test-name text.
- `apply_tick_windows`: ~16–24 px hollow square inside the red tick strip; if sheet-wide grid drops >0.02 vs `snap_overlay`, fall back to snap bboxes.
- Synthetic hollow **≥ 95%**.

```mermaid
flowchart TB
  bars[Printed headers]
  rows[Printed row peaks]
  ticks[Tick strip minus label]
  sq[Hollow square per field_id]
  bars --> rows --> ticks --> sq
```

---

## B1.5 — 1a / 1b split (2026-08-28)

**Facts:** `block1-1a-1b-split`, `block1-no-mark-classification`. Session `2026-08-28-144429`.

**Architecture**

| Stage | Module | Does |
|---|---|---|
| 1a | `block1a.py` | Revision pick, warp, page `fine_align`, `col_shifts` |
| 1b | `block1b.py` | `snap_overlay` → bake col shifts → `snap_sections` → per-section **dy** (peaks vs **row tops**) → tick squares (prefer **ink-in-ring**) → extra-ink `crops/sections/{id}.png` |

Square placement may fall back to dy-shifted snap if tick-window grid is >0.02 worse; **extra-ink always keeps section lock**. Block 1 does **not** classify marks (`mark_classification: deferred_to_block3`).

```mermaid
flowchart LR
  a[1a warp align]
  b[1b sections squares extra-ink]
  zip[Block 1 ZIP]
  a --> b --> zip
```

**Data then:** Pytest `tests/test_normalization.py` 25 passed, synthetic hollow ≥ 95%.

---

## B1.6 — 1c crop gate (2026-08-31) — **current**

**Fact:** `block1-1c-crop-validate`. Code: `block1c.py`, wired in `pipeline.py`.

**Architecture** (does not classify ticks)

1. If the crop is already a printed square (empty hollow or ink-in-ring) and not a text line → `crop_ok`.
2. Else one widened hollow / ink-ring rematch: reject neighbour-steal and dark section headers.
3. Empty fields rematch onto **hollow** rings only (do not steal a neighbour’s ink).
4. Else keep the 1b bbox and set `crop_needs_hitl`.
5. Large overlay-sized cells skip 1c unless they look like a label.

ZIP checkbox metadata: `crop_ok`, `crop_needs_hitl`, `crop_validate_status` (`ok` / `retry` / `hitl` / `skip`). Block 3 ORs `crop_needs_hitl` into `needs_hitl`.

```mermaid
flowchart LR
  photo[Photo]
  a[1a warp]
  b[1b squares]
  c[1c window gate]
  b3[Block 3]
  photo --> a --> b --> c --> b3
```

### Current data (B1.6 + Block 3 B3.5, Paddle off, 2026-08-31)

Synthetic v0 blank `data/samples/synthetic/lab_request_v0_blank.png`, 10 gold ticks drawn on template squares, optional `photograph()` tilt. **No clinic PHI.** Full JSON: [data/synthetic-10tick-scorecard.json](./data/synthetic-10tick-scorecard.json).

| Sheet | 1c summary | Block 3 TP / FN / FP |
|---|---|---|
| Blank digital | retry 111, skip 13, hitl 0 | 0 / — / **0** |
| 10 ticks digital | ok 10, retry 105, skip 9 | **10 / 0 / 0** |
| Blank photo | retry 113, skip 11 | 0 / — / **3** |
| 10 ticks photo | ok 9, retry 105, skip 10 | **4 / 6 / 5** |

Pre-1c / pre-geometry Block 3 on the same 10 digital ticks was **0 / 10 / 0**.

Committed inputs (no PHI):

- `data/samples/synthetic/lab_request_v0_blank.png`
- `templates/lab_request_canonical.json` (v0)
- `templates/lab_request_v1_canonical.json` (v1)

---

## What later Block 1 versions must not do

- Tick classification (Block 3).
- Unbounded 1c rematch loops.
- Committing clinic PNGs.

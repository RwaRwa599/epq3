# Block 1 section-bar snap experiment (2026-08-20)

Gated overlay snap using printed column section bars (HAEMATOLOGY, URINE, …), not matching 138 identical checkbox rings. Hollow-square benchmark vs the previous ring-only snap. Clinic photos were processed locally; this report has no PHI.

## What was implemented

After warp + fine-align, `snap_overlay` still does the existing **hollow-ring** snap. It also tries a **section-bar** snap:

1. Split fields into the four gutter columns.
2. Detect black header bars in each column (full-width dark bands).
3. Pair them in order with expected bars from template `group` (`haematology`, `bone_nutrition`, `cardiovascular`, `tumour_markers`, `urine`, `stool`, `immunology`, `endocrinology`).
4. Fit a per-column Y affine `y' = a·y + b` (reject if max residual > 18 px or |b| is huge).
5. Shift that column’s overlay, then snap at most one hollow square per row.

The section result is **kept only if** `_checkbox_grid_score` improves by at least 0.02. Otherwise the old ring snap wins. That is the “if worse, don’t use it” gate inside a single document.

Test-name strip matching and letter-initial matching (`r,l,f,n`) were prototyped and **not** shipped (see below).

## Hollow benchmark

Pass line for synthetic: ≥ 95% of checkbox crops look like a hollow square (`center > 210`, `0.04 ≤ dark_frac ≤ 0.65`).

| Setup | Before (ring snap) | After (gated section bars) |
|---|---|---|
| Rendered canonical v0 | 118/124 = **95.2%** | **95.2%** |
| Photographed + warped v0 | 118/124 = **95.2%** | **95.2%** |
| Digital v0 blank PNG | 90/124 = 72.6% | 72.6% (metric: thin rings, not mis-snap) |
| Pytest `tests/test_normalization.py` | 11 passed | 13 passed |

Clinic five-sheet (auto v1, current `block1` warp):

| Sheet | Before | After | Snap method used |
|---|---|---|---|
| s1 | 137/138 = 99.3% | 99.3% | rings (section tied, gate not taken) |
| s2 | 95/138 = 68.8% | 68.8% | rings |
| s3 | 38/138 = 27.5% | 44/138 = **31.9%** | **section-bars** (2 columns fitted) |
| s4 | 44/138 = 31.9% | 31.9% | section-bars (grid up, hollow flat) |
| s5 | 61/138 = 44.2% | 44.2% | rings (section grid +1.4 pp, below 2 pp gate) |
| **Mean** | **54.3%** | **55.2%** | |

Synthetic still passes. Clinic mean is **not worse** (+0.9 pp). The change is kept with the gate. Ungated section affine (always apply) was worse than rings and was **not** kept.

## What was tried and dropped

**1. Match printed test names (NCC / ink profile of the label strip)**  
Text has more stable ink than empty boxes, so this was the intended locator. On clinic sheets, pairing each expected row to the **nearest** ink peak is circular: when Y is already wrong, the nearest peak is the wrong test. Mean hollow **fell** (e.g. s1 99.3% → 87% if forced). Initials (`Renal Function` → r,l,f,n) were not implemented; they collide and are not cheaper than matching a whole black bar.

**2. Uniform per-column Y from text correlation**  
Same issue: locks onto a wrong peak at the search edge (±70–90 px). Per-column “only keep if that column’s hollow score rises” almost never fired, because a pure Y translation does not fix a sheared full-frame warp.

**3. Ungated section-bar affine + local square**  
Bar pairing on col0/col3 is **geometrically good** when ≥2 bars are visible (s1 residuals ~0.4 px; s3 col0 `a=1.05`, `b≈-109`, residual 1.6 px). Applying it anyway **lowered** hollow on several sheets (s2 68.8% → 65.2%, s5 44.2% → 40.6%). Headers can sit on the right Y while checkbox **X** is still off, so crops miss the square. Extra bars (footer / OTHERS) can 1-1 pair to the wrong consecutive window if residual is a median; the shipped fit uses **max** residual to reject that.

## Analysis

Section bars are a good **identity** landmark: unique, high-contrast, almost never written on. They verify that a column’s overlay is in the right neighbourhood. They are a weak **crop** landmark: the tick is a 20 px square whose X still comes from a full-frame homography. On s3 the bars say “shift this column up ~100 px”; doing that improves grid score enough to pass the gate, but most crops still fail the hollow test because the remaining error is warp (full-frame stretch), not missing headers.

That is why s1 does not change (already on-grid) and s3–s5 stay far below 95%. Label-to-the-right matching will not beat that until the page quad is closer, or until each row is locked by a **unique** printed name inside a **small** band after the section bar has put that band on the right block.

## Decision

- **Keep** gated section-bar snap in `src/med_doc/normalization/align.py` (`snap_overlay` method `section-bars` only when grid_score ≥ ring_score + 0.02).
- **Do not** replace ring snap with whole-page name matching or letter initials.
- Next lever for the hollow benchmark is **warp** (full-frame vs a real page quad), not more overlay heuristics on a sheared canvas.

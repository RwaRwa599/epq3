# Changelog (branch `block1`)

Live Colab: [Run_in_Colab.ipynb](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/Run_in_Colab.ipynb) — expect `BOOTSTRAP_V6` and `tick_policy: slash-v2`.

Full accuracy tables and “why this dump failed → next commit”: [`docs/blocks/clinic-eval-chain.md`](docs/blocks/clinic-eval-chain.md). Architecture generations: [`docs/blocks/block1-versions.md`](docs/blocks/block1-versions.md), [`docs/blocks/block3-versions.md`](docs/blocks/block3-versions.md).

| When | Git / policy | Clinic result (tiny N, photos not in git) | Why next version |
|---|---|---|---|
| B3.4 | Paddle on ticks | 7–43 ticks/sheet; 10 TP / 10 FN / 41 FP (other workspace) | Glyphs ≠ ticks |
| B3.5 | Interior geometry | Synthetic digital 10-tick **P=1.00 R=1.00**; photo **P=0.44 R=0.40** | Still needs Block 1 windows |
| order-2/3 | Logreg + metadata | ~21 / ~55 FPs; missed CA125; invented EDTA | No logreg-only ticks |
| `d8c1a68` | Slash + profile HiTL | User order-5 still expanded profiles | Stale Colab → BOOTSTRAP_V6 |
| `9af79a9` slash-v2 | order-7 IMG_7598 | **P=0 R=0** (only ALP FP; 7 gold missed) | Bound Y-shift one row |
| `80233a4` | order-8 | **R=0**, empty order | 1b on labels; 1c skipped ink-in-ring |
| `786156f` + **Block 5 gate** | order-9 / order-10 | order-9 **P=0.20 R≈0.29**; order-10 **19–37 ticks/sheet** | Rematch trusted wrong windows |
| `block1` B1.8 | handwriting 1c + align gate | Phase 1: tubes no longer on NT-proBNP; `<0.6` align → `needs_review`; skip not silent-ok | Phase 2 ECC / neighbour fill; Block 5 date hallucination |

Verbal (synthetic): tubes **1.00 / 1.00**, dates **1.00 / 0.50**, others **1.00 / 1.00** (clean / photoreal).

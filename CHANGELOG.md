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
| `block1` B1.9 | ECC + decoupled page gate | Page `<0.6` still `needs_review` but 1c/3 keep per-field scores; relative template margin 10%; implausible tubes trip registration_failure | Clinic photos still need ECC/piecewise to lift 0.36–0.54 |
| `block1` B1.11 | header-row template audit + v1 blank for HTR | Renal extra row / Urea≠Na row; others charset soup from v0 blank residual | Label-neighbor check on `canonical.png` when present |
| `block1` B3.10 | 3c VL draft + Block 4 KG rank + Instruct n-best | default off; VLM ticks HiTL-only; disagreement crops capped | Cloud VLMs; auto-union ticks from co-occurrence |
| `block1` if1 | Parallel if1block1–4 + `prototype3.ipynb` (`BOOTSTRAP_V8`) | Live 1–5 unchanged; Bayesian group fit is a placeholder | Counts for \(w,\pi\); two-tier LabOrder |

Verbal (synthetic): tubes **1.00 / 1.00**, dates **1.00 / 0.50**, others **1.00 / 1.00** (clean / photoreal).

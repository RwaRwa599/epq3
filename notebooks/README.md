# Colab notebooks (live `src/med_doc`)

Same input shapes as Block 1a: a **folder**, a **ZIP of photos**, a **list of paths**, or Colab multi-upload. The runner is `run_blocks_1_to_5`.

Do **not** upload clinic PHI. The demos use `data/samples/synthetic/lab_request_v0_blank.png`.

The pipeline notebook runs **`run_blocks_1_to_5`** on a folder, ZIP, or multi-file upload (same input shapes as Block 1a). Default **`output_mode="user"`** writes one Block 5 file, `order.json`. Set `OUTPUT_MODE = "dev"` for per-block ZIPs.

| Notebook | Open in Colab |
|---|---|
| Full pipeline (**use this**) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/Run_in_Colab.ipynb) |
| Block 1 — normalize | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_1_Document_Normalization.ipynb) |
| Block 2 — KG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_2_Knowledge_Graph.ipynb) |
| Block 3 — marks + HTR drafts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_3_Marks_and_HTR.ipynb) |
| Block 4 — KG rescoring | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_4_KG_Rescoring.ipynb) |
| Block 5 — review + LIS | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_5_Review_and_LIS.ipynb) |

First cell prints **`BOOTSTRAP_V6`** plus **`tick_policy: slash-v2`** and **always** re-downloads `https://codeload.github.com/RwaRwa599/epq3/zip/refs/heads/block1` (a leftover `/content/epq3` is the usual cause of fake ticks / missing `output_mode`). No git clone and no `%pip install -e`. If imports fail: **Runtime → Disconnect and delete runtime**, then open `Run_in_Colab.ipynb` from GitHub branch `block1`.

The `block1/`, `block2/`, `block3/` package trees are older snapshots; their notebooks still point at snapshot branches.

Version history (accuracy + why each dump failed): [`docs/blocks/clinic-eval-chain.md`](../docs/blocks/clinic-eval-chain.md).

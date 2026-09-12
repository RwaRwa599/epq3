# Colab notebooks (live `src/med_doc`)

These notebooks clone branch **`block1`**. Opening an `.ipynb` from GitHub does **not** download `src/`. Use **Runtime → Run all**, or run the first code cell before any `import med_doc`. Each import cell also puts `src/` on `sys.path` (Colab `pip install -e .` often does not see the package until restart).

Do **not** upload clinic PHI. The demos use `data/samples/synthetic/lab_request_v0_blank.png`.

| Notebook | Open in Colab |
|---|---|
| Full pipeline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/Pipeline_Blocks_1_to_5.ipynb) |
| Block 1 — normalize | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_1_Document_Normalization.ipynb) |
| Block 2 — KG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_2_Knowledge_Graph.ipynb) |
| Block 3 — marks + HTR drafts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_3_Marks_and_HTR.ipynb) |
| Block 4 — KG rescoring | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_4_KG_Rescoring.ipynb) |
| Block 5 — review + LIS | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RwaRwa599/epq3/blob/block1/notebooks/Block_5_Review_and_LIS.ipynb) |

Private repo: add Colab secret `GITHUB_TOKEN`. The setup cell uses it for `git clone`.

The `block1/`, `block2/`, `block3/` package trees are older snapshots; their notebooks still point at snapshot branches.

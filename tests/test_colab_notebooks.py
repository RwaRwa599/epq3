"""Colab notebooks must clone the live tree (GitHub open does not include src/)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "Pipeline_Blocks_1_to_5.ipynb",
    ROOT / "notebooks" / "Block_1_Document_Normalization.ipynb",
    ROOT / "notebooks" / "Block_2_Knowledge_Graph.ipynb",
    ROOT / "notebooks" / "Block_3_Marks_and_HTR.ipynb",
    ROOT / "notebooks" / "Block_4_KG_Rescoring.ipynb",
    ROOT / "notebooks" / "Block_5_Review_and_LIS.ipynb",
]


def test_colab_notebooks_clone_live_branch():
    for path in NOTEBOOKS:
        assert path.exists(), path
        text = path.read_text(encoding="utf-8")
        assert "RwaRwa599/epq3.git" in text
        assert "block1" in text
        assert "PHI" in text or "phi" in text.lower()
        assert "BOOTSTRAP_V3" in text or "bootstrap()" in text
        assert "sys.path.insert" in text
        assert "med_doc" in text

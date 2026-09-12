from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "Run_in_Colab.ipynb",
    ROOT / "Pipeline_Blocks_1_to_5.ipynb",
    ROOT / "notebooks" / "Block_1_Document_Normalization.ipynb",
    ROOT / "notebooks" / "Block_2_Knowledge_Graph.ipynb",
    ROOT / "notebooks" / "Block_3_Marks_and_HTR.ipynb",
    ROOT / "notebooks" / "Block_4_KG_Rescoring.ipynb",
    ROOT / "notebooks" / "Block_5_Review_and_LIS.ipynb",
]


def test_colab_notebooks_zipball_bootstrap_v5():
    for path in NOTEBOOKS:
        assert path.exists(), path
        text = path.read_text(encoding="utf-8")
        assert "BOOTSTRAP_V5" in text, path.name
        assert "output_mode" in text
        assert "codeload.github.com/RwaRwa599/epq3/zip/refs/heads/block1" in text
        assert "block1" in text
        assert "PHI" in text or "phi" in text.lower()
        assert "sys.path.insert" in text
        assert "med_doc" in text
        assert "%pip install -q -e" not in text
        assert "pip install -e" not in text
        assert "git clone" not in text


def test_colab_bootstrap_from_local_zip(tmp_path, monkeypatch):
    import zipfile
    import sys

    sys.path.insert(0, str(ROOT / "notebooks"))
    from colab_bootstrap import bootstrap_med_doc  # type: ignore

    zip_path = tmp_path / "tree.zip"
    pkg = ROOT / "src" / "med_doc"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in pkg.rglob("*"):
            if file.is_file() and "__pycache__" not in file.parts:
                rel = file.relative_to(ROOT)
                zf.write(file, arcname=f"epq3-block1/{rel.as_posix()}")

    dest_parent = tmp_path / "content"
    dest_parent.mkdir()
    workspace_src = str((ROOT / "src").resolve())
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(workspace_src)]
    sys.modules.pop("med_doc", None)

    root = bootstrap_med_doc(content=dest_parent, url=zip_path.resolve().as_uri())
    assert (root / "src" / "med_doc" / "__init__.py").is_file()
    import med_doc

    assert Path(med_doc.__file__).resolve().is_relative_to(dest_parent.resolve())

    stale = dest_parent / "epq3" / "src" / "med_doc" / "stale_marker.txt"
    stale.write_text("old", encoding="utf-8")
    root2 = bootstrap_med_doc(content=dest_parent, url=zip_path.resolve().as_uri())
    assert not (root2 / "src" / "med_doc" / "stale_marker.txt").exists()

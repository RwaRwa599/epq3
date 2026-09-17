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
    ROOT / "notebooks" / "Layout_Crop_Batch.ipynb",
]

PROTOTYPE3 = ROOT / "prototype3.ipynb"


def test_colab_notebooks_zipball_bootstrap_v6():
    for path in NOTEBOOKS:
        assert path.exists(), path
        text = path.read_text(encoding="utf-8")
        assert "BOOTSTRAP_V6" in text, path.name
        assert "slash-v2" in text, path.name
        assert "output_mode" in text
        assert "codeload.github.com/RwaRwa599/epq3/zip/refs/heads/block1" in text
        assert "block1" in text
        assert "PHI" in text or "phi" in text.lower()
        assert "sys.path.insert" in text
        assert "med_doc" in text
        assert "%pip install -q -e" not in text
        assert "pip install -e" not in text
        assert "git clone" not in text


def test_prototype3_if1_zipball_bootstrap_v9():
    assert PROTOTYPE3.exists(), PROTOTYPE3
    text = PROTOTYPE3.read_text(encoding="utf-8")
    assert "BOOTSTRAP_V9" in text
    assert "slash-v2" in text
    assert "PIPELINE" in text
    assert "run_blocks_1_to_5" in text
    assert "if1=True" in text or "if1=(PIPELINE" in text
    assert "ordered_tests_high" in text
    assert "ordered_tests_low" in text
    assert "codeload.github.com/RwaRwa599/epq3/zip/refs/heads/block1" in text
    assert "if1block1.zip" in text
    assert "block5.zip" in text
    assert "PHI" in text or "phi" in text.lower()
    assert "sys.path.insert" in text
    assert "med_doc" in text
    assert "%pip install -q -e" not in text
    assert "pip install -e" not in text
    assert "git clone" not in text
    assert "PIPELINE = \\\"if1\\\"" in text or 'PIPELINE = "if1"' in text


def test_colab_bootstrap_from_local_zip(tmp_path, monkeypatch):
    import os
    import zipfile
    import sys

    cwd = os.getcwd()
    path0 = list(sys.path)
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

    try:
        root = bootstrap_med_doc(content=dest_parent, url=zip_path.resolve().as_uri())
        assert (root / "src" / "med_doc" / "__init__.py").is_file()
        import med_doc

        assert Path(med_doc.__file__).resolve().is_relative_to(dest_parent.resolve())

        stale = dest_parent / "epq3" / "src" / "med_doc" / "stale_marker.txt"
        stale.write_text("old", encoding="utf-8")
        root2 = bootstrap_med_doc(content=dest_parent, url=zip_path.resolve().as_uri())
        assert not (root2 / "src" / "med_doc" / "stale_marker.txt").exists()
    finally:
        os.chdir(cwd)
        sys.path[:] = path0
        for name in list(sys.modules):
            if name == "med_doc" or name.startswith("med_doc."):
                del sys.modules[name]
        sys.path.insert(0, str((ROOT / "src").resolve()))

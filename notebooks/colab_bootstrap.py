"""Colab/kernel bootstrap for live src/med_doc.

Opening a GitHub notebook does not clone the repo. Run this as the first cell.
Prints BOOTSTRAP_V3 when import med_doc succeeds.
"""

from __future__ import annotations

import os
import site
import subprocess
import sys
from pathlib import Path

BOOTSTRAP_VERSION = "BOOTSTRAP_V3"
REPO = "https://github.com/RwaRwa599/epq3.git"
BRANCH = "block1"


def _run(cmd: list[str]) -> None:
    print("$", " ".join(str(c) for c in cmd))
    subprocess.check_call(cmd)


def _token() -> str | None:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        return tok
    try:
        from google.colab import userdata

        return userdata.get("GITHUB_TOKEN")
    except Exception:
        return None


def _clone_url() -> str:
    tok = _token()
    if tok:
        return f"https://{tok}@github.com/RwaRwa599/epq3.git"
    return REPO


def _find_root() -> Path | None:
    here = Path.cwd().resolve()
    for cand in (
        here,
        here.parent,
        Path("/content/epq3"),
        Path("/content") / "epq3",
    ):
        if (cand / "src" / "med_doc" / "__init__.py").is_file():
            return cand
    return None


def _write_pth(src: Path) -> None:
    line = str(src.resolve()) + "\n"
    dirs = []
    try:
        dirs.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        dirs.append(site.getusersitepackages())
    except Exception:
        pass
    sp = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    dirs.append(str(sp))
    for d in dirs:
        if not d:
            continue
        target = Path(d)
        try:
            target.mkdir(parents=True, exist_ok=True)
            (target / "epq3_src.pth").write_text(line, encoding="utf-8")
            print("wrote", target / "epq3_src.pth")
        except Exception as exc:
            print("pth skip", target, exc)


def _ipython_cd(path: Path) -> None:
    try:
        ip = get_ipython()  # type: ignore[name-defined]
    except Exception:
        ip = None
    if ip is None:
        os.chdir(path)
        return
    ip.run_line_magic("cd", str(path))


def _ipython_pip(root: Path) -> None:
    try:
        ip = get_ipython()  # type: ignore[name-defined]
    except Exception:
        ip = None
    if ip is not None:
        ip.run_line_magic("pip", "install -q matplotlib opencv-python-headless pydantic Pillow numpy")
        ip.run_line_magic("pip", f"install -q -e {root}")
        return
    _run([sys.executable, "-m", "pip", "install", "-q", "matplotlib", "opencv-python-headless", "pydantic", "Pillow", "numpy"])
    _run([sys.executable, "-m", "pip", "install", "-q", "-e", str(root)])


def put_src_on_path(root: Path | None = None) -> Path:
    root = root or _find_root()
    if root is None:
        raise ModuleNotFoundError(
            "med_doc not found. Run the first notebook cell (BOOTSTRAP_V3 clone). "
            "Private repo: Colab secret GITHUB_TOKEN. Then Runtime → Run all."
        )
    src = (root / "src").resolve()
    os.chdir(root)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.environ["PYTHONPATH"] = str(src) + os.pathsep + os.environ.get("PYTHONPATH", "")
    return root


def bootstrap() -> Path:
    print(BOOTSTRAP_VERSION)
    dest = Path("/content/epq3") if Path("/content").is_dir() else (Path.cwd().resolve() / "epq3")
    root = _find_root()
    if root is None:
        url = _clone_url()
        if dest.exists() and not (dest / "src" / "med_doc" / "__init__.py").is_file():
            import shutil

            shutil.rmtree(dest, ignore_errors=True)
        if not (dest / ".git").is_dir():
            _run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--single-branch", url, str(dest)])
        else:
            _run(["git", "-C", str(dest), "fetch", "origin", BRANCH])
            _run(["git", "-C", str(dest), "checkout", BRANCH])
            _run(["git", "-C", str(dest), "pull", "--ff-only", "origin", BRANCH])
        root = dest
    _ipython_cd(root)
    os.chdir(root)
    src = root / "src"
    if str(src.resolve()) not in sys.path:
        sys.path.insert(0, str(src.resolve()))
    _write_pth(src)
    try:
        _ipython_pip(root)
    except Exception as exc:
        print("pip note:", exc)
    # Drop a copy next to cwd as last resort (some Colab kernels ignore .pth until restart)
    try:
        import med_doc  # noqa: F401
    except ModuleNotFoundError:
        sys.path.insert(0, str(src.resolve()))
        import importlib

        importlib.invalidate_caches()
        import med_doc  # noqa: F401
    import med_doc

    print("cwd:", os.getcwd())
    print("med_doc:", med_doc.__file__)
    if "src" not in Path(med_doc.__file__).parts:
        print("warning: unexpected med_doc location")
    return root


if __name__ == "__main__":
    bootstrap()

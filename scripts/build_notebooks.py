"""Convert notebooks/src_py/*.py ('# %%' cell scripts) into .ipynb and execute them.

Usage: python scripts/build_notebooks.py [name ...]   (default: all, in order)
Each notebook runs top to bottom with the repo root as working directory;
a failing cell fails the build.
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "src_py"


def to_nb(text: str) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "ipo-predictor", "display_name": "Python (ipo-predictor)", "language": "python"}
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if not body:
            return
        if kind == "md":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(body))
        else:
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("md" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    nb.cells = cells
    return nb


def main(names):
    files = sorted(SRC.glob("*.py"))
    if names:
        files = [f for f in files if any(n in f.stem for n in names)]
    for f in files:
        nb = to_nb(f.read_text(encoding="utf-8"))
        print(f"executing {f.stem} ...", flush=True)
        NotebookClient(nb, timeout=3600, kernel_name="ipo-predictor", resources={"metadata": {"path": str(ROOT)}}).execute()
        out = ROOT / "notebooks" / f"{f.stem}.ipynb"
        nbformat.write(nb, out)
        print(f"  wrote {out.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

"""Convert a percent-format script (# %% / # %% [markdown]) into a .ipynb.

The .py file is the reviewed source; the .ipynb is generated from it:
  python synthetic-study/tools/py2nb.py synthetic-study/notebooks/00_setup.py
"""
import json
import re
import sys
from pathlib import Path

MARK = re.compile(r"^# %%(?P<md> \[markdown\])?(?P<title>.*)$")


def convert(src: Path) -> Path:
    cells, cur, is_md = [], [], False

    def flush():
        lines = cur[:]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            return
        if is_md:
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": _join(lines)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": _join(lines)})

    for line in src.read_text(encoding="utf-8").splitlines():
        m = MARK.match(line)
        if m:
            flush()
            cur, is_md = [], bool(m["md"])
            title = m["title"].strip()
            if title and not is_md:
                cur.append(f"# {title}")
            continue
        cur.append(line)
    flush()
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:02d}"
    out = src.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def _join(lines):
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(convert(Path(arg)))

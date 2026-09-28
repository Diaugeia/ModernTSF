"""tsf init — scaffold a standalone project that uses an installed ModernTSF.

The generated project owns only its run configs, data, and outputs. Catalog
configs (``base.toml``, datasets, models) are inherited from the installed
package through ``moderntsf://`` extends paths, so upgrading ModernTSF
upgrades the defaults without copying them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

FILES: dict[str, str] = {
    "configs/runs/example.toml": """\
# Inherit catalog defaults from the installed ModernTSF package.
extends = [
    "moderntsf://configs/base.toml",
    "moderntsf://configs/datasets/etth1.toml",
    "moderntsf://configs/models/DLinear.toml",
]

[experiment]
description = "{name}: DLinear on ETTh1"

[sweep.task]
pred_len = [96, 192]
""",
    "dataset/README.md": """\
# dataset/

Local data files referenced by `configs/`. Inspect the catalog with
`tsf dataset list` and `tsf dataset show <name>`; ETTh1 is expected at
`dataset/ETT-small/ETTh1.csv`.
""",
    ".gitignore": """\
dataset/*
!dataset/README.md
work_dirs/
__pycache__/
.venv/
""",
    "pyproject.toml": """\
[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["modern-tsf[hub]"]
""",
    "README.md": """\
# {name}

A time-series forecasting project built on
[ModernTSF](https://github.com/Diaugeia/ModernTSF).

```bash
uv sync                                    # install modern-tsf[hub]
tsf env                                    # check devices, data, and output capacity
tsf inspect --config configs/runs/example.toml   # preview the expanded runs
tsf run configs/runs/example.toml          # train and evaluate; outputs go to work_dirs/
tsf hub push <run_id> --repo <owner>/<weights-repo>   # publish a checkpoint (explicit)
```

Run configs extend the installed catalog with `moderntsf://configs/...`
paths; local `extends` paths work as usual.
""",
}


def init_project(target: Path, name: str | None = None, force: bool = False) -> list[Path]:
    """Write the project skeleton into ``target`` and return the created files."""
    name = name or target.resolve().name
    existing = [target / rel for rel in FILES if (target / rel).exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite: " + ", ".join(str(path) for path in existing)
        )
    written = []
    for rel, template in FILES.items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template.replace("{name}", name), encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tsf init", description=__doc__)
    parser.add_argument("directory", nargs="?", default=".")
    parser.add_argument("--name", help="project name (default: directory name)")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args(argv)
    try:
        written = init_project(Path(args.directory), args.name, args.force)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for path in written:
        print(f"created {path}")
    return 0

#!/usr/bin/env python3
"""Create an unregistered model workspace from resolved paper/source facts.

Use ``tsf model scaffold`` only after paper extraction and component matching.
The placeholder is not added to the catalog. After implementing it, adding
manifest checks, and completing the card, run ``tsf model add --name``.
"""

from __future__ import annotations

import argparse
import re
import sys

from tsflab.tsf_core.paths import repository_root, require_checkout


ROOT = repository_root()
MODELS_DIR = ROOT / "src" / "tsflab" / "models"
MODEL_CONFIG_DIR = ROOT / "configs" / "models"
RUN_CONFIG_DIR = ROOT / "configs" / "runs"
_PY_DEFAULTS = {"int", "float", "str", "bool"}


def _module_slug(name: str) -> str:
    """Derive the flat lowercase Python package slug for a public model name."""
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def _package_init(name: str) -> str:
    """Return the explicit package entrypoint for an unregistered workspace."""
    return f'"""Local {name} model package."""\n\nfrom .model import Model\n\n__all__ = ["Model"]\n'


def _parse_params(spec: str | None) -> list[tuple[str, str, str | None]]:
    result: list[tuple[str, str, str | None]] = []
    for chunk in (spec or "").split(","):
        if not chunk.strip():
            continue
        field, separator, rest = chunk.strip().partition(":")
        if not separator or not field.isidentifier():
            raise SystemExit(f"invalid parameter declaration: {chunk!r}")
        kind, equals, default = rest.partition("=")
        kind = kind.strip()
        if kind not in _PY_DEFAULTS:
            raise SystemExit(f"unsupported parameter type {kind!r}")
        result.append((field, kind, default.strip() if equals else None))
    return result


def _literal(kind: str, value: str | None) -> str:
    if value is None:
        return {"int": "128", "float": "0.1", "str": '"value"', "bool": "true"}[kind]
    if kind == "bool":  # TOML booleans are lower-case whatever the caller wrote
        return "true" if value.strip().lower() in {"true", "1", "yes"} else "false"
    return value if kind != "str" else f'"{value.strip(chr(34))}"'


def _python_literal(kind: str, value: str) -> str:
    literal = _literal(kind, value)
    return literal.replace("true", "True").replace("false", "False")


def _schema(name: str, params: list[tuple[str, str, str | None]], graph: bool) -> str:
    lines = [
        f'"""Validated parameters for {name}."""', "",
        "from pydantic import BaseModel, ConfigDict", "", "",
        "class ModelParameterConfig(BaseModel):",
        '    """Parameters supplied through ``model.params``."""', "",
        '    model_config = ConfigDict(extra="forbid")', "",
        "    enc_in: int",
    ]
    if graph:
        lines.append("    cov_dim: int = 2")
    for field, kind, default in params:
        if field in {"enc_in", "cov_dim"}:
            continue
        declaration = f"    {field}: {kind}"
        if default is not None:
            declaration += f" = {_python_literal(kind, default)}"
        lines.append(declaration)
    return "\n".join(lines) + "\n"


def _spec(
    name: str,
    module: str,
    params: list[tuple[str, str, str | None]],
    task_mode: str,
    components: tuple[str, ...],
) -> str:
    graph = task_mode != "time_series"
    arguments = [
        "        seq_len=cfg.task.seq_len,",
        "        pred_len=cfg.task.pred_len,",
        '        enc_in=params["enc_in"],',
    ]
    if graph:
        arguments.extend(
            ('        adj_mx=params.get("adj_mx"),', '        cov_dim=params.get("cov_dim", 2),')
        )
    for field, kind, default in params:
        if field in {"enc_in", "cov_dim"}:
            continue
        access = (
            f'params["{field}"]'
            if default is None
            else f'params.get("{field}", {_python_literal(kind, default)})'
        )
        arguments.append(f"        {field}={access},")
    capability = {
        "time_series": "time-series",
        "spatiotemporal": "spatiotemporal",
        "covariate": "covariate",
    }[task_mode]
    arguments_text = "\n".join(arguments)
    return f'''"""Runtime specification for {name}."""

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.{module}.model import Model

{_schema(name, params, graph)}

def build_model(cfg, params):
    return Model(
{arguments_text}
    )


SPEC = ModelSpec(
    name="{name}",
    module="tsflab.models.{module}",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/{name}.toml",
    model_card="src/tsflab/models/{module}/README.md",
    smoke_config="configs/runs/smoke_{module}.toml",
    capabilities=frozenset({{{capability!r}}}),
    components={components!r},
    contract_task={{"seq_len": {24 if graph else 96}, "pred_len": {24 if graph else 12}, "label_len": 0}},
)
'''


def _card(
    name: str,
    paper_title: str,
    paper_url: str,
    venue: str,
    year: int,
    codebase: tuple[str, str, str] | None,
    components: tuple[str, ...],
) -> str:
    source = ""
    if codebase:
        url, revision, license_name = codebase
        source = (
            f'code: "{url}"\n'
            f'revision: "{revision}"\n'
            f'license: "{license_name}"\n'
        )
    component_text = ", ".join(f"`{item}`" for item in components) or "none"
    return f'''---
name: "{name}"
summary: "SCAFFOLD: replace with an evidence-backed method summary."
paper: "{paper_url}"
paper_title: "{paper_title}"
venue: "{venue}"
year: {year}
{source}
---
# {name}

<!-- model-card:canonical:start -->
## Input and output

Document the four-input forecasting interface and exact output semantics.

## Paper and code

Explain which paper and pinned official-code details were checked.

## Local implementation

Map paper equations to `model.py`; do not copy external model source.

## Differences

Record every material difference from the paper or official implementation.

## Shared components

Planned components: {component_text}. Keep only verified imports.

## Configuration constraints

Document shape, parameter, optional-input, and artifact requirements.
<!-- model-card:canonical:end -->
'''


def _model(name: str, params: list[tuple[str, str, str | None]], graph: bool) -> str:
    extras = []
    for field, kind, default in params:
        if field in {"enc_in", "cov_dim"}:
            continue
        suffix = "" if default is None else f" = {_python_literal(kind, default)}"
        extras.append(f"        {field}: {kind}{suffix},")
    extra_text = "\n".join(extras)
    graph_imports = "import numpy as np\n\nfrom tsflab.models._components.marks import to_spatiotemporal\n" if graph else ""
    graph_args = '        adj_mx: "np.ndarray | None" = None,\n        cov_dim: int = 2,\n' if graph else ""
    setup = (
        "        if adj_mx is None:\n"
        "            adj_mx = np.eye(enc_in, dtype=np.float32)\n"
        "        self.register_buffer(\"adj_mx\", torch.as_tensor(adj_mx, dtype=torch.float32))\n"
        "        self.cov_dim = cov_dim\n"
        if graph else ""
    )
    forward = (
        "        if x_mark_enc is None:\n"
        "            x_mark_enc = x_enc.new_zeros((x_enc.shape[0], x_enc.shape[1], 6))\n"
        "        values = to_spatiotemporal(x_enc, x_mark_enc)[..., 0]\n"
        "        return self.placeholder(values.transpose(1, 2)).transpose(1, 2)\n"
        if graph else
        "        return self.placeholder(x_enc.transpose(1, 2)).transpose(1, 2)\n"
    )
    return f'''"""SCAFFOLD for the local {name} implementation; replace before admission."""

from __future__ import annotations

{graph_imports}import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
{graph_args}{extra_text}
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
{setup}        self.placeholder = nn.Linear(seq_len, pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
{forward}'''


def _model_config(name: str, params: list[tuple[str, str, str | None]], graph: bool) -> str:
    lines = ["[model]", f'name = "{name}"', "", "[model.params]", f"enc_in = {8 if graph else 7}"]
    if graph:
        lines.append("cov_dim = 2")
    for field, kind, default in params:
        if field not in {"enc_in", "cov_dim"}:
            lines.append(f"{field} = {_literal(kind, default)}")
    return "\n".join(lines) + "\n"


def _smoke(name: str, module: str, task_mode: str) -> str:
    graph = task_mode != "time_series"
    dataset = "../fixtures/synthetic_st.toml" if graph else "../fixtures/smoke.toml"
    return f'''extends = ["../base.toml", "{dataset}", "../models/{name}.toml"]

[experiment]
description = "Smoke: {name}"

[experiment.runtime]
device = "cpu"
num_workers = 0

[task]
mode = "{task_mode}"
seq_len = {24 if graph else 96}
label_len = 0
pred_len = {24 if graph else 12}
features = "M"

[training]
epochs = 1
batch_size = 8
loss = "mae"
patience = 1

[model.params]
enc_in = {8 if graph else 6}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--module")
    parser.add_argument("--params")
    parser.add_argument("--paper-title", required=True)
    parser.add_argument("--paper-url", required=True)
    parser.add_argument("--venue", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--code-url")
    parser.add_argument("--revision")
    parser.add_argument("--license")
    parser.add_argument("--components", required=True, help="comma-separated names or 'none'")
    parser.add_argument(
        "--task-mode", choices=["time_series", "spatiotemporal", "covariate"],
        default="time_series",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        require_checkout("tsf model scaffold")
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from None

    source_values = (args.code_url, args.revision, args.license)
    if any(source_values) and not all(source_values):
        parser.error("--code-url, --revision, and --license must be supplied together")
    codebase = source_values if all(source_values) else None
    components = () if args.components.strip().lower() == "none" else tuple(
        item.strip() for item in args.components.split(",") if item.strip()
    )
    from tsflab.benchmark.catalog.components import COMPONENT_CATALOG

    if len(set(components)) != len(components):
        parser.error("--components contains duplicates")
    unknown_components = sorted(set(components) - set(COMPONENT_CATALOG.names()))
    if unknown_components:
        parser.error(
            "unknown component(s): " + ", ".join(unknown_components)
            + "; run 'tsf component match <terms>' first"
        )
    if args.task_mode != "time_series" and "marks" not in components:
        parser.error("node/covariate scaffold uses the shared 'marks' component")

    name = args.name
    module = args.module or _module_slug(name)
    params = _parse_params(args.params)
    graph = args.task_mode != "time_series"
    package = MODELS_DIR / module
    targets = {
        package / "__init__.py": _package_init(name),
        package / "model.py": _model(name, params, graph),
        package / "spec.py": _spec(name, module, params, args.task_mode, components),
        package / "README.md": _card(
            name, args.paper_title, args.paper_url, args.venue, args.year,
            codebase, components,
        ),
        MODEL_CONFIG_DIR / f"{name}.toml": _model_config(name, params, graph),
        RUN_CONFIG_DIR / f"smoke_{module}.toml": _smoke(name, module, args.task_mode),
    }
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        print("Refusing to overwrite existing files:", file=sys.stderr)
        for path in existing:
            print(f"  {path.relative_to(ROOT)}", file=sys.stderr)
        raise SystemExit(1)
    package.mkdir(parents=True, exist_ok=True)
    for path, content in targets.items():
        path.write_text(content, encoding="utf-8")

    print(f"Created unregistered model workspace {name!r} ({module})")
    for path in targets:
        print(f"  + {path.relative_to(ROOT)}")
    print("Next: replace SCAFFOLD code/card text, add manifest tests, then run")
    print(f"  tsf model add --name {name}")


if __name__ == "__main__":
    main()

"""Generate and audit canonical README cards for datasets and models._components."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
import tomllib

from moderntsf.benchmark.catalog.component_audit import components_used_by
from moderntsf.benchmark.catalog.components import COMPONENT_CATALOG, ComponentSpec


@dataclass(frozen=True)
class DatasetRecord:
    """Resolved metadata for one runnable dataset configuration preset."""

    name: str
    config: str
    loader: str
    alias: str
    path: str
    dataset_id: str
    params: dict[str, object]
    task: dict[str, object]
    track: str
    task_modes: tuple[str, ...]


def _quoted(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _task_modes_for_loader(loader: str) -> tuple[str, ...]:
    """Resolve modes from the executable registry instead of duplicating them."""
    from moderntsf.benchmark.registry.datasets import (
        DATASET_REGISTRY,
        ordered_task_modes,
        register_dataset_by_name,
    )

    register_dataset_by_name(loader)
    return ordered_task_modes(DATASET_REGISTRY.get(loader).task_modes)


def dataset_records(root: Path) -> tuple[DatasetRecord, ...]:
    """Read every dataset preset without importing its runtime dependencies."""
    config_root = root / "configs" / "datasets"
    records: list[DatasetRecord] = []
    for path in sorted(config_root.rglob("*.toml")):
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        dataset = payload.get("dataset", {})
        relative = path.relative_to(config_root)
        name = relative.with_suffix("").as_posix()
        records.append(
            DatasetRecord(
                name=name,
                config=path.relative_to(root).as_posix(),
                loader=str(dataset.get("name", "")),
                alias=str(dataset.get("alias", name)),
                path=str(dataset.get("path", "")),
                dataset_id=str(dataset.get("id", "")),
                params=dict(dataset.get("params", {})),
                task=dict(payload.get("task", {})),
                track=str(dataset.get("track", "")),
                task_modes=_task_modes_for_loader(str(dataset.get("name", ""))),
            )
        )
    return tuple(records)


def component_card_path(root: Path, name: str) -> Path:
    """Return the canonical component-card path."""
    return root / "src" / "moderntsf" / "models" / "_components" / name / "README.md"


def dataset_card_path(root: Path, name: str) -> Path:
    """Return the canonical dataset-card path."""
    return root / "catalog" / "datasets" / Path(name) / "README.md"


def _component_consumers(root: Path, name: str) -> tuple[str, ...]:
    consumers = []
    for package in sorted((root / "src" / "moderntsf" / "models").iterdir()):
        if package.is_dir() and not package.name.startswith("_") and name in components_used_by(package):
            consumers.append(package.name)
    return tuple(consumers)


def _first_paragraph(value: str | None) -> str:
    """Compact a docstring to its descriptive opening paragraph."""
    if not value:
        return "No additional symbol-level description is recorded."
    return " ".join(value.strip().split("\n\n", 1)[0].split())


def _component_api(root: Path, spec: ComponentSpec) -> tuple[str, str]:
    """Read module and public-symbol documentation without importing code."""
    path = root / "src" / "moderntsf" / "models" / "_components" / spec.name / "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    lines = []
    for symbol in spec.public_symbols:
        node = nodes.get(symbol)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signature = f"{symbol}({ast.unparse(node.args)})"
            detail = _first_paragraph(ast.get_docstring(node))
        elif isinstance(node, ast.ClassDef):
            initializer = next(
                (
                    child
                    for child in node.body
                    if isinstance(child, ast.FunctionDef) and child.name == "__init__"
                ),
                None,
            )
            arguments = ast.unparse(initializer.args) if initializer else ""
            if arguments.startswith("self, "):
                arguments = arguments[6:]
            elif arguments == "self":
                arguments = ""
            signature = f"{symbol}({arguments})"
            detail = _first_paragraph(ast.get_docstring(node))
        else:
            signature = symbol
            detail = "Public module constant."
        lines.extend((f"- `{signature}`", f"  {detail}"))
    if not lines:
        lines.append("- Import the module and use its documented functions/classes.")
    return _first_paragraph(ast.get_docstring(tree)), "\n".join(lines)


def render_component_card(root: Path, spec: ComponentSpec) -> str:
    """Render a component card from its catalog contract and real consumers."""
    consumers = _component_consumers(root, spec.name)
    module_description, symbols = _component_api(root, spec)
    consumer_lines = "\n".join(
        f"- [`{name}`](../../{name}/README.md)" for name in consumers
    )
    if not consumer_lines:
        consumer_lines = "- No model currently declares this component directly."
    import_hint = (
        f"from {spec.module} import {', '.join(spec.public_symbols)}"
        if spec.public_symbols
        else f"import {spec.module}"
    )
    keywords = ", ".join(f"`{keyword}`" for keyword in spec.keywords)
    return f"""---
name: {_quoted(spec.name)}
kind: "component"
module: {_quoted(spec.module)}
summary: {_quoted(spec.contract)}
---

# {spec.name}

## Purpose

{spec.contract}

{module_description}

Implementation: [`__init__.py`](__init__.py)

## Public API

{symbols}

```python
{import_hint}
```

## Input and output contract

Tensor axes, accepted values, validation rules, and returned shapes are defined by
the public symbol docstrings and runtime checks in the implementation. Preserve
those semantics when composing the component; matching tensor rank alone is not
sufficient.

## Composition guidance

Retrieve this component with `tsf component match`, inspect this card and its
implementation, then declare `{spec.name}` in the consuming model's `components`
tuple. The repository audit checks that declaration against actual imports.

Retrieval terms: {keywords}.

## Current model consumers

{consumer_lines}

## Semantic boundary

This card documents one reusable contract, not a promise that similarly named
model-local blocks are interchangeable. Keep a block model-local when its axis
meaning, normalization, state update, or paper equation differs.
"""


def _dataset_mode(record: DatasetRecord) -> tuple[tuple[str, ...], str, str]:
    """Return supported task modes, summary, and item contract for a preset."""
    if record.loader == "gift_eval":
        horizon = record.task.get("pred_len", "configured")
        return (
            record.task_modes,
            f"GIFT-Eval preset for {record.dataset_id!r} with forecast horizon {horizon}.",
            "Windowed history/target values and timestamp marks; after batching, values use `[batch, time, channels]`.",
        )
    if record.loader in {"cauair_st", "synthetic_st"}:
        return (
            record.task_modes,
            f"Node-structured spatiotemporal preset loaded by `{record.loader}`.",
            "Each item is `(value_history, value_future, covariate_history, covariate_future)`; values use `[time, nodes]` and covariates `[time, nodes, features]` before batching.",
        )
    if record.loader == "cauair_ts":
        return (
            record.task_modes,
            "CauAir-style node data flattened to a multivariate time-series preset.",
            "Each item contains history/future values shaped `[time, nodes]` plus six-column zero timestamp marks before batching.",
        )
    return (
        record.task_modes,
        f"Time-series forecasting preset loaded by `{record.loader}`.",
        "Each item provides history/target windows and timestamp marks; after batching, values use `[batch, time, channels]`.",
    )


def render_dataset_card(record: DatasetRecord) -> str:
    """Render a dataset card directly from its executable TOML preset."""
    task_modes, summary, item_contract = _dataset_mode(record)
    task = json.dumps(record.task, ensure_ascii=False, indent=2, sort_keys=True)
    params = json.dumps(record.params, ensure_ascii=False, indent=2, sort_keys=True)
    track = record.track or "standard"
    root_prefix = "../" * (3 + record.name.count("/"))
    return f"""---
name: {_quoted(record.name)}
kind: "dataset"
config: {_quoted(record.config)}
loader: {_quoted(record.loader)}
alias: {_quoted(record.alias)}
task_modes: {json.dumps(task_modes, ensure_ascii=False)}
summary: {_quoted(summary)}
---

# {record.alias}

## Overview

{summary} This card describes the repository preset and runtime contract; it
does not add an external-source provenance claim that is absent from the configuration.

## Loader and files

- Registry loader: `{record.loader}`
- Config: [`{record.config}`]({root_prefix}{record.config})
- Local path: `{record.path or '(loader-defined)'}`
- Dataset id: `{record.dataset_id or '(not applicable)'}`
- Track: `{track}`

## Input and output contract

{item_contract}

Sequence length, label length, feature mode, and batch size are supplied by the
experiment task unless explicitly overridden below.

## Dataset parameters

```json
{params}
```

## Task overrides

```json
{task}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config {record.config}` and
prepare/download data with the loader-specific dataset command when required.
Reference this preset from an experiment configuration rather than duplicating
its loader parameters.

## Composition constraints

Choose one of `{', '.join(task_modes)}` and match it to the model's declared task
mode. Inspect the loader
before changing feature or scaling parameters. Paths are repository defaults and
may need local overrides; the card does not imply that the data is bundled.
"""


def expected_resource_cards(root: Path) -> dict[Path, str]:
    """Return every canonical generated card and its expected content."""
    expected = {
        component_card_path(root, spec.name): render_component_card(root, spec)
        for spec in COMPONENT_CATALOG.specs()
    }
    expected.update(
        {
            dataset_card_path(root, record.name): render_dataset_card(record)
            for record in dataset_records(root)
        }
    )
    return expected


def write_resource_cards(root: Path) -> int:
    """Write all canonical cards and return their count."""
    expected = expected_resource_cards(root)
    for path, content in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return len(expected)


def audit_resource_cards(root: Path) -> list[str]:
    """Report missing, stale, or orphaned generated resource cards."""
    expected = expected_resource_cards(root)
    errors = []
    for path, content in expected.items():
        if not path.is_file():
            errors.append(f"missing resource card: {path.relative_to(root)}")
        elif path.read_text(encoding="utf-8") != content:
            errors.append(f"stale resource card: {path.relative_to(root)}")
    actual = set((root / "src" / "moderntsf" / "models" / "_components").glob("*/README.md"))
    actual.update((root / "catalog" / "datasets").glob("**/README.md"))
    for path in sorted(actual - set(expected)):
        errors.append(f"orphaned resource card: {path.relative_to(root)}")
    return errors

"""Generate and audit canonical README cards for datasets and models._components."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from moderntsf.benchmark.catalog.component_audit import components_used_by
from moderntsf.benchmark.catalog.components import COMPONENT_CATALOG, ComponentSpec
from moderntsf.benchmark.dataset_cards import (  # noqa: F401  (re-exported public names)
    DatasetRecord,
    audit_dataset_cards,
    dataset_card_path,
    dataset_records,
    expected_dataset_cards,
    write_dataset_cards,
)


def _quoted(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def component_card_path(root: Path, name: str) -> Path:
    """Return the canonical component-card path."""
    return root / "src" / "moderntsf" / "models" / "_components" / name / "README.md"


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


def expected_resource_cards(root: Path) -> dict[Path, str]:
    """Return every generated component card and its expected content.

    Dataset cards mix curated and generated content, so their expectation
    depends on the file on disk; see ``dataset_cards.expected_dataset_cards``.
    """
    expected = {
        component_card_path(root, spec.name): render_component_card(root, spec)
        for spec in COMPONENT_CATALOG.specs()
    }
    return expected


def write_resource_cards(root: Path) -> int:
    """Write all canonical cards and return their count."""
    expected = expected_resource_cards(root)
    for path, content in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return len(expected) + write_dataset_cards(root)


def audit_resource_cards(root: Path) -> list[str]:
    """Report missing, stale, or orphaned generated resource cards."""
    expected = expected_resource_cards(root)
    errors = []
    for path, content in expected.items():
        if not path.is_file():
            errors.append(f"missing resource card: {path.relative_to(root)}")
        elif path.read_text(encoding="utf-8") != content:
            errors.append(f"stale resource card: {path.relative_to(root)}")
    errors.extend(audit_dataset_cards(root))
    expected_paths = set(expected) | set(expected_dataset_cards(root))
    actual = set((root / "src" / "moderntsf" / "models" / "_components").glob("*/README.md"))
    actual.update((root / "catalog" / "datasets").glob("**/README.md"))
    for path in sorted(actual - expected_paths):
        errors.append(f"orphaned resource card: {path.relative_to(root)}")
    return errors

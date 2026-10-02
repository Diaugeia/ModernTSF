"""Curated component cards with a generated reference block, and their audit.

A component card is ``README.md`` beside the component code. Its front matter
and curated sections are written by hand from the code, tests, and consumers;
only the block between the generated markers is rendered from the catalog and
the source (public API, import line, retrieval terms, consumers).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re

from tsflab.benchmark.card_depth import read_card, split_sections
from tsflab.benchmark.catalog.component_audit import components_used_by
from tsflab.benchmark.catalog.components import COMPONENT_CATALOG, ComponentSpec

START = "<!-- component-card:generated:start -->"
END = "<!-- component-card:generated:end -->"

CATEGORIES = (
    "attention",
    "backbone",
    "convolution",
    "decomposition",
    "embedding",
    "frequency",
    "fusion",
    "graph",
    "head",
    "memory",
    "mixer",
    "normalization",
    "routing",
    "state-space",
    "utility",
)
FRONT_MATTER = (
    "name",
    "kind",
    "module",
    "summary",
    "category",
    "input",
    "output",
    "origin",
    "origin_models",
    "tags",
)
SECTIONS = (
    "Purpose",
    "Origin and granularity",
    "Interface",
    "Invariants and equivalence evidence",
    "Variants and options",
    "When to use and when not to use",
    "Related components",
)
_PLACEHOLDER = re.compile(r"\b(?:TODO|TBD|FIXME)\b|\bto be written\b", re.IGNORECASE)
_PATH = re.compile(r"`((?:tests|src|verification|configs)/[^`\s]+)`")
_MIN_SECTION_CHARS = 40


def component_card_path(root: Path, name: str) -> Path:
    """Return the canonical component-card path."""
    return root / "src" / "tsflab" / "models" / "_components" / name / "README.md"


def _quoted(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _first_paragraph(value: str | None) -> str:
    if not value:
        return "No symbol-level description is recorded."
    return " ".join(value.strip().split("\n\n", 1)[0].split())


def _public_api(root: Path, spec: ComponentSpec) -> str:
    """Read public-symbol signatures and docstrings without importing code."""
    path = root / "src" / "tsflab" / "models" / "_components" / spec.name / "__init__.py"
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
    return "\n".join(lines)


def _consumers(root: Path, name: str) -> tuple[str, ...]:
    return tuple(
        package.name
        for package in sorted((root / "src" / "tsflab" / "models").iterdir())
        if package.is_dir()
        and not package.name.startswith("_")
        and name in components_used_by(package)
    )


def render_generated_block(root: Path, spec: ComponentSpec) -> str:
    """Render the machine-owned reference block for one component card."""
    consumers = _consumers(root, spec.name)
    consumer_text = (
        ", ".join(f"`{name}`" for name in consumers)
        if consumers
        else "No model currently declares this component directly."
    )
    import_hint = (
        f"from {spec.module} import {', '.join(spec.public_symbols)}"
        if spec.public_symbols
        else f"import {spec.module}"
    )
    keywords = ", ".join(f"`{keyword}`" for keyword in spec.keywords)
    return f"""{START}
## Public API

Implementation: [`__init__.py`](__init__.py)

{_public_api(root, spec)}

```python
{import_hint}
```

## Retrieval terms

{keywords}

## Current model consumers ({len(consumers)})

{consumer_text}
{END}"""


def _skeleton(spec: ComponentSpec) -> str:
    """Return a new card whose curated fields the audit rejects until filled."""
    tags = json.dumps(list(spec.keywords))
    sections = "\n\n".join(f"## {title}\n\nTODO" for title in SECTIONS)
    return f"""---
name: {_quoted(spec.name)}
kind: "component"
module: {_quoted(spec.module)}
summary: {_quoted(spec.contract)}
category: "TODO"
input: "TODO"
output: "TODO"
origin: "TODO"
origin_models: []
tags: {tags}
---

# {spec.name}

{sections}

{START}
{END}
"""


def update_component_card(root: Path, spec: ComponentSpec) -> bool:
    """Refresh the generated block, creating a skeleton for a new component."""
    path = component_card_path(root, spec.name)
    text = path.read_text(encoding="utf-8") if path.is_file() else _skeleton(spec)
    block = render_generated_block(root, spec)
    if START in text and END in text:
        head, rest = text.split(START, 1)
        updated = head + block + rest.split(END, 1)[1]
    else:
        updated = text.rstrip() + "\n\n" + block + "\n"
    if path.is_file() and updated == path.read_text(encoding="utf-8"):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")
    return True


def audit_component_card(root: Path, spec: ComponentSpec) -> list[str]:
    """Return every contract violation of one component card."""
    path = component_card_path(root, spec.name)
    label = path.relative_to(root).as_posix()
    if not path.is_file():
        return [f"missing component card: {label}"]
    try:
        card = read_card(path)
    except ValueError as exc:
        return [f"{label}: {exc}"]
    errors: list[str] = []
    front = card.front
    for key in FRONT_MATTER:
        if key not in front:
            errors.append(f"{label}: front matter missing {key!r}")
    if errors:
        return errors
    if front["name"] != spec.name or front["kind"] != "component":
        errors.append(f"{label}: front matter name/kind must be {spec.name!r}/'component'")
    if front["module"] != spec.module:
        errors.append(f"{label}: front matter module must be {spec.module!r}")
    for key in ("summary", "input", "output", "origin"):
        value = front[key]
        if not isinstance(value, str) or not value.strip() or _PLACEHOLDER.search(value):
            errors.append(f"{label}: front matter {key!r} must be a real, non-empty string")
    summary = str(front["summary"])
    if len(summary) > 200:
        errors.append(f"{label}: summary exceeds 200 characters; keep it one L0 line")
    if front["category"] not in CATEGORIES:
        errors.append(f"{label}: category must be one of {', '.join(CATEGORIES)}")
    tags = front["tags"]
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        errors.append(f"{label}: tags must be a list of strings")
    else:
        if len(set(tags)) != len(tags):
            errors.append(f"{label}: duplicate tags")
        missing = sorted(set(spec.keywords) - set(tags))
        if missing:
            errors.append(f"{label}: tags must include catalog keywords {missing}")
    origin_models = front["origin_models"]
    models_root = root / "src" / "tsflab" / "models"
    if not isinstance(origin_models, list):
        errors.append(f"{label}: origin_models must be a list of model slugs")
    else:
        for slug in origin_models:
            if not (models_root / str(slug) / "model.py").is_file():
                errors.append(f"{label}: origin model {slug!r} is not a model package")

    if card.text.count(START) != 1 or card.text.count(END) != 1:
        errors.append(f"{label}: needs exactly one generated block")
    else:
        actual = START + card.text.split(START, 1)[1].split(END, 1)[0] + END
        if actual != render_generated_block(root, spec):
            errors.append(f"stale component card: {label}")

    curated = card.text.split(START, 1)[0]
    sections = dict(split_sections(curated))
    titles = [title for title, _ in split_sections(curated)]
    if titles != list(SECTIONS):
        errors.append(f"{label}: curated sections must be exactly, in order: {', '.join(SECTIONS)}")
    for title in SECTIONS:
        body = sections.get(title, "")
        if len(body) < _MIN_SECTION_CHARS or _PLACEHOLDER.search(body):
            errors.append(f"{label}: section {title!r} is empty or a placeholder")
    interface = sections.get("Interface", "")
    for symbol in spec.public_symbols:
        if symbol not in interface:
            errors.append(f"{label}: Interface does not describe public symbol {symbol!r}")
    for target in sorted(set(_PATH.findall(curated))):
        if not (root / target.rstrip(".,;:")).exists():
            errors.append(f"{label}: referenced path does not exist: {target}")
    evidence = sections.get("Invariants and equivalence evidence", "")
    if not _PATH.search(evidence) and "no fixture" not in evidence.lower():
        errors.append(
            f"{label}: evidence must cite an existing tests/ or fixture path, "
            "or state 'no fixture' with the reason"
        )
    return errors


def audit_component_cards(root: Path) -> list[str]:
    """Audit every cataloged component card and report orphans."""
    errors: list[str] = []
    for spec in COMPONENT_CATALOG.specs():
        errors.extend(audit_component_card(root, spec))
    actual = {
        path.parent.name
        for path in (root / "src" / "tsflab" / "models" / "_components").glob("*/README.md")
    }
    for name in sorted(actual - set(COMPONENT_CATALOG.names())):
        errors.append(
            f"orphaned resource card: src/tsflab/models/_components/{name}/README.md"
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    """Refresh generated blocks and audit the named (default: all) component cards."""
    import sys

    from tsflab.tsf_core.paths import require_checkout

    root = require_checkout("component-card regeneration")
    names = list(sys.argv[1:] if argv is None else argv) or COMPONENT_CATALOG.names()
    errors: list[str] = []
    for name in names:
        spec = COMPONENT_CATALOG.get(name)
        update_component_card(root, spec)
        errors.extend(audit_component_card(root, spec))
    for error in errors:
        print(f"ERROR: {error}")
    print(f"component cards: {len(names) - len({e.split(':')[0] for e in errors})}/{len(names)} OK")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

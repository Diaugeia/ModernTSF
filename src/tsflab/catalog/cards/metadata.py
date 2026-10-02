"""Torch-free readers for runtime specs and canonical model-card metadata."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import TypedDict, cast


class PaperMetadata(TypedDict):
    title: str
    venue: str
    year: int | None
    url: str


class CodebaseMetadata(TypedDict):
    url: str
    revision: str
    license: str


def _literal(node: ast.expr):
    """Read literal values and frozensets used by runtime specifications."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "frozenset":
            return tuple(ast.literal_eval(node.args[0])) if node.args else ()
    return ast.literal_eval(node)


def declared_model_fields(spec_file: Path) -> dict[str, object]:
    """Read literal top-level ``ModelSpec`` runtime fields without imports."""
    tree = ast.parse(spec_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "SPEC" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Call):
            return {}
        fields: dict[str, object] = {}
        for keyword in node.value.keywords:
            if keyword.arg is None:
                continue
            try:
                fields[keyword.arg] = _literal(keyword.value)
            except (ValueError, TypeError):
                continue
        return fields
    return {}


def _scalar(value: str) -> object:
    """Parse the deliberately small scalar subset used by model cards."""
    value = value.strip()
    if not value or value in {"null", "~"}:
        return None
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith(('"', "'")):
        return json.loads(value) if value.startswith('"') else value[1:-1]
    if value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    try:
        return int(value)
    except ValueError:
        return value


def read_front_matter(path: Path) -> dict[str, object]:
    """Read the deliberately small YAML-like front matter without PyYAML."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path} has no YAML front matter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError(f"{path} has unterminated YAML front matter") from exc
    result: dict[str, object] = {}
    section: dict[str, object] | None = None
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in line:
            raise ValueError(f"{path} has invalid front-matter line: {line!r}")
        key, raw = line.strip().split(":", 1)
        if indent == 0:
            if raw.strip():
                result[key] = _scalar(raw)
                section = None
            else:
                nested: dict[str, object] = {}
                result[key] = nested
                section = nested
        elif indent == 2 and section is not None:
            section[key] = _scalar(raw)
        else:
            raise ValueError(f"{path} has unsupported front-matter indentation")
    return result


def read_model_card(path: Path) -> dict[str, object]:
    """Validate the flat human header and return normalized metadata.

    Model cards keep a short, flat header for people.  Runtime consumers receive
    normalized ``paper`` and ``codebase`` mappings so generated indexes and
    verification evidence do not duplicate parsing logic.
    """
    fields = read_front_matter(path)
    required = {"name", "summary", "paper", "paper_title", "venue", "year"}
    allowed = required | {"code", "revision", "license", "tagline", "tags", "composition"}
    missing = sorted(required - fields.keys())
    if missing:
        raise ValueError(f"{path} missing front matter: {', '.join(missing)}")
    unexpected = sorted(fields.keys() - allowed)
    if unexpected:
        raise ValueError(f"{path} has unsupported front matter: {', '.join(unexpected)}")
    source_fields = {key for key in ("code", "revision", "license") if key in fields}
    if source_fields and source_fields != {"code", "revision", "license"}:
        missing_source = sorted({"code", "revision", "license"} - source_fields)
        raise ValueError(f"{path} missing code fields: {', '.join(missing_source)}")
    if "code" in fields and not str(fields["code"] or "").strip():
        raise ValueError(f"{path} front matter code must not be empty")
    return {
        "name": fields["name"],
        "summary": fields["summary"],
        "tagline": fields.get("tagline", ""),
        "tags": list(fields.get("tags") or []),
        "composition": list(fields.get("composition") or []),
        "paper": {
            "title": fields["paper_title"],
            "venue": fields["venue"],
            "year": fields["year"],
            "url": fields["paper"],
        },
        "codebase": (
            {
                "url": fields["code"],
                "revision": fields["revision"],
                "license": fields["license"],
            }
            if "code" in fields
            else None
        ),
    }


def model_records(
    root: Path, refs: dict[str, str] | None = None
) -> list[dict[str, object]]:
    """Return records for the registered catalog only.

    A scaffold may already contain a valid-looking spec and card while it is
    still being implemented. Registration, not filesystem presence, is the
    admission boundary, so unregistered workspaces must never leak into CLI
    discovery, generated docs, or verification.
    """
    if refs is None:
        from tsflab.catalog.registry.models import MODEL_CATALOG

        refs = MODEL_CATALOG.refs()
    records: list[dict[str, object]] = []
    for registered_name, module_path in refs.items():
        path = root / "src" / Path(*module_path.split(".")).with_suffix(".py")
        if not path.is_file():
            raise ValueError(
                f"registered model {registered_name!r} is missing {path}"
            )
        runtime = declared_model_fields(path)
        if not runtime:
            raise ValueError(
                f"registered model {registered_name!r} has no literal ModelSpec"
            )
        card_path = path.parent / "README.md"
        metadata = read_model_card(card_path)
        if (
            runtime.get("name") != registered_name
            or metadata.get("name") != registered_name
        ):
            raise ValueError(
                f"registered model {registered_name!r} disagrees with "
                f"{path.relative_to(root)} or its card"
            )
        fields = {**runtime, **metadata}
        fields["package"] = path.parent.name
        fields["spec_file"] = str(path.relative_to(root))
        fields["model_card"] = str(card_path.relative_to(root))
        records.append(fields)
    return sorted(records, key=lambda record: str(record["name"]).casefold())


def paper_metadata(record: dict[str, object]) -> PaperMetadata:
    """Return the typed paper mapping from a model record."""
    return cast(PaperMetadata, record["paper"])


def codebase_metadata(record: dict[str, object]) -> CodebaseMetadata | None:
    """Return the typed codebase mapping from a model record."""
    return cast(CodebaseMetadata | None, record["codebase"])

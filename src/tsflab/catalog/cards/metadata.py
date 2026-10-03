"""Torch-free readers for runtime specs and canonical model-card metadata."""

from __future__ import annotations

import ast
import json
from pathlib import Path


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


def read_model_card(path: Path) -> dict[str, object]:
    """Return normalized metadata for a model card (``card.toml`` + README description).

    ``path`` is the card's README or its directory. Runtime consumers receive the
    same flat shape as before (``summary``, ``tagline``, ``tags``, ``composition``
    as ``slot=value`` strings, ``paper``, ``codebase``) plus the card's ``fits``,
    ``fidelity``, ``data_params``, ``issues``, and ``admission``.
    """
    from tsflab.catalog.cards.store import load

    card = load(path if path.is_dir() else path.parent)
    facts = card.facts
    if facts.get("kind") != "model":
        raise ValueError(f"{card.directory}/card.toml is not a model card")
    paper = facts.get("paper") or {}
    code = facts.get("code")
    return {
        "name": facts["name"],
        "summary": card.description,
        "tagline": card.description,
        "tags": list(facts.get("tags") or []),
        "composition": [f"{slot}={value}" for slot, value in (facts.get("composition") or {}).items()],
        "paper": {
            "title": paper.get("title", ""),
            "venue": paper.get("venue", ""),
            "year": paper.get("year"),
            "url": paper.get("url", ""),
        },
        "codebase": (
            {"url": code["url"], "revision": code["revision"], "license": code["license"]}
            if code else None
        ),
        "fits": list(facts.get("fits") or []),
        "fidelity": facts.get("fidelity", ""),
        "data_params": dict(facts.get("data_params") or {}),
        "issues": list(facts.get("issues") or []),
        "admission": dict(facts.get("admission") or {}),
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



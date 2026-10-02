"""Validate human documentation projections, links, and audience boundaries.

Catches the exact kind of drift that's easy to introduce by hand and easy to
miss in review: stale generated model pages, unindexed guides, and Agent-only
implementation details leaking into user documentation.
Pure text/file checks keep this safe and fast enough for every PR.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from tsflab.tsf_core.paths import repository_root, require_checkout

from tsflab.benchmark.catalog_metadata import model_records

ROOT = repository_root()
DOC_DIR = ROOT / "docs" / "en"
OBSOLETE_DOC_TOKENS = (
    "moderntsf",  # pre-rename package name
    "tool/",
    "modern-tsf",
    "MODEL_NAME_MAP",
    "MODEL_REGISTRY",
    "registry.py",
    "schema.py",
    "src/tsflab/models/module",
    "src/tsflab/models/_external",
)
AGENT_ONLY_TOKENS = (
    ".agents/",
    ".claude/skills",
    "AGENTS.md",
    "CLAUDE.md",
    "SKILL.md",
    "tsflab.benchmark.commands",
)


def render_models_doc() -> str:
    records = model_records(ROOT)
    intro = (
        "# Models and methods\n\n"
        f"TSFLab exposes {len(records)} model and method entries through one flat "
        "public catalog. There are no user-facing architecture families. Presets "
        "configure runs and do not create additional entries.\n\n"
        "Every entry is maintained as a local implementation; verification status "
        "is derived from executable evidence.\n\n"
        "| Name | Preset | Capabilities | Model card |\n"
        "|---|---|---|---|\n"
    )
    rows = []
    for record in records:
        name = str(record["name"])
        config = str(record["config_path"])
        package = str(record["package"])
        capabilities = ", ".join(sorted(record.get("capabilities", ()))) or "—"
        rows.append(
            f"| `{name}` | [`{config}`](../../{config}) | {capabilities} | "
            f"[README](../../src/tsflab/models/{package}/README.md) |"
        )
    return intro + "\n".join(rows) + "\n"


def _generated_model_doc_problems() -> list[str]:
    problems: list[str] = []
    path = DOC_DIR / "models.md"
    if path.read_text(encoding="utf-8") != render_models_doc():
        problems.append(f"{path.relative_to(ROOT)} is stale; regenerate from ModelSpec")
    return problems


def write_generated_model_docs() -> None:
    """Refresh the human-readable model catalog projection."""
    (DOC_DIR / "models.md").write_text(render_models_doc(), encoding="utf-8")


def _docs_index_problems() -> list[str]:
    problems = []
    readme = DOC_DIR / "README.md"
    readme_text = readme.read_text()
    linked = set(re.findall(r"\[([a-zA-Z0-9_.-]+\.md)\]", readme_text))
    for page in DOC_DIR.glob("*.md"):
        if page.name == "README.md":
            continue
        if page.name not in linked:
            problems.append(
                f"{page.relative_to(ROOT)} has no link in {readme.relative_to(ROOT)}"
            )
    return problems


def _obsolete_reference_problems() -> list[str]:
    problems: list[str] = []
    paths = [ROOT / "README.md", *DOC_DIR.glob("*.md")]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in OBSOLETE_DOC_TOKENS:
            if token in text:
                problems.append(f"{path.relative_to(ROOT)} references obsolete {token!r}")
    return problems


def _audience_boundary_problems() -> list[str]:
    problems: list[str] = []
    human_paths = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *DOC_DIR.glob("*.md")]
    for path in human_paths:
        text = path.read_text(encoding="utf-8")
        for token in AGENT_ONLY_TOKENS:
            if token in text:
                problems.append(
                    f"{path.relative_to(ROOT)} exposes Agent-only detail {token!r}"
                )

    for path in (ROOT / ".agents").rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for token in ("docs/en/", "CONTRIBUTING.md"):
            if token in text:
                problems.append(
                    f"{path.relative_to(ROOT)} depends on human documentation {token!r}"
                )
    return problems


def _relative_link_problems() -> list[str]:
    problems: list[str] = []
    paths = [
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "CHANGELOG.md",
        ROOT / "THIRD_PARTY_NOTICES.md",
    ]
    paths.extend(DOC_DIR.glob("*.md"))
    paths.extend((ROOT / ".agents").rglob("*.md"))
    paths.extend((ROOT / "src" / "tsflab" / "models").glob("*/README.md"))

    for path in paths:
        text = path.read_text(encoding="utf-8")
        for raw_target in re.findall(r"\]\(([^)]+)\)", text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith(("#", "/", "mailto:")):
                continue
            resolved = path.parent / target
            if not resolved.exists():
                problems.append(
                    f"{path.relative_to(ROOT)} has broken relative link {raw_target!r}"
                )
    return problems


def check() -> list[str]:
    return (
        _generated_model_doc_problems()
        + _docs_index_problems()
        + _obsolete_reference_problems()
        + _audience_boundary_problems()
        + _relative_link_problems()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write",
        action="store_true",
        help="regenerate the model catalog before checking",
    )
    args = parser.parse_args()
    if args.write:
        require_checkout("documentation regeneration")
        write_generated_model_docs()
    problems = check()
    if not problems:
        print("OK: generated model docs and indexes are consistent.")
        return 0
    print(f"Found {len(problems)} doc inconsistency(ies):")
    for p in problems:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

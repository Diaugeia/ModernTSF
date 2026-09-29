"""Validate the repository's cross-harness Agent-first asset layout."""

from __future__ import annotations

import re
from pathlib import Path

from moderntsf.tsf_core.paths import is_packaged_root, repository_root


ROOT = repository_root()
SKILLS = ROOT / ".agents" / "skills"
STANDARDS = ROOT / ".agents" / "STANDARDS.md"
EXPECTED_SKILLS = {
    "add-dataset",
    "add-model",
    "analyze-results",
    "audit-model",
    "audit-repository",
    "curate-components",
    "design-experiment",
    "diagnose-experiment",
    "discover-papers",
    "expand-model-catalog",
    "extract-paper-structure",
    "forecast-realtime-round",
    "implement-model",
    "inspect-dataset",
    "integrate-foundation-model",
    "prepare-dataset",
    "publish-weights",
    "report-defect",
    "reproduce-paper-results",
    "run-experiment",
    "run-autoresearch",
    "setup-environment",
    "smoke-models",
    "submit-results",
}


def _link_target(path: Path) -> str | None:
    return str(path.readlink()) if path.is_symlink() else None


def audit_agent_assets() -> list[str]:
    """Return violations of the cross-harness Agent-first contract."""
    errors: list[str] = []
    agents_md = ROOT / "AGENTS.md"
    claude_md = ROOT / "CLAUDE.md"
    claude_skills = ROOT / ".claude" / "skills"

    if not agents_md.is_file() or agents_md.is_symlink():
        errors.append("AGENTS.md must be a real file")
    elif len(agents_md.read_text(encoding="utf-8").splitlines()) > 50:
        errors.append("AGENTS.md exceeds the 50-line always-loaded context budget")
    if not is_packaged_root(ROOT) and _link_target(claude_md) != "AGENTS.md":
        errors.append("CLAUDE.md must be a symlink to AGENTS.md")
    if not SKILLS.is_dir() or SKILLS.is_symlink():
        errors.append(".agents/skills must be the real canonical skill directory")
    if not STANDARDS.is_file() or STANDARDS.is_symlink():
        errors.append(".agents/STANDARDS.md must be the real consolidated contract")
    elif len(STANDARDS.read_text(encoding="utf-8").splitlines()) > 100:
        errors.append(".agents/STANDARDS.md exceeds the 100-line on-demand budget")
    if not is_packaged_root(ROOT) and _link_target(claude_skills) != "../.agents/skills":
        errors.append(".claude/skills must link to ../.agents/skills")
    for duplicate_root in (ROOT / ".pi" / "skills", ROOT / ".dsh" / "skills"):
        if duplicate_root.exists() or duplicate_root.is_symlink():
            errors.append(
                f"{duplicate_root.relative_to(ROOT)} duplicates native .agents/skills discovery"
            )
    obsolete_agent_docs = (
        ROOT / ".agents" / "README.md",
        ROOT / ".agents" / "HARNESS_COMPATIBILITY.md",
        ROOT / ".agents" / "standards",
    )
    for obsolete_doc in obsolete_agent_docs:
        if obsolete_doc.exists() or obsolete_doc.is_symlink():
            errors.append(
                f"{obsolete_doc.relative_to(ROOT)} duplicates .agents/STANDARDS.md"
            )
    root_agent_docs = {path.name for path in (ROOT / ".agents").glob("*.md")}
    if root_agent_docs != {"STANDARDS.md"}:
        errors.append(".agents may contain only the consolidated STANDARDS.md at its root")

    for skill_file in sorted(SKILLS.rglob("SKILL.md")):
        if skill_file.parent.parent != SKILLS:
            errors.append(
                f"{skill_file.relative_to(ROOT)}: skills must be one level below .agents/skills"
            )

    seen: set[str] = set()
    for skill_file in sorted(SKILLS.glob("*/SKILL.md")):
        text = skill_file.read_text(encoding="utf-8")
        if len(text.splitlines()) > 80:
            errors.append(
                f"{skill_file.relative_to(ROOT)}: exceeds the 80-line entrypoint budget"
            )
        match = re.match(r"^---\n(?P<header>.*?)\n---\n", text, re.DOTALL)
        if match is None:
            errors.append(f"{skill_file.relative_to(ROOT)}: missing YAML frontmatter")
            continue
        header = match.group("header")
        name_match = re.search(r"^name:\s*[\"']?([^\"'\n]+)", header, re.MULTILINE)
        desc_match = re.search(
            r"^description:\s*(?P<description>.+?)\s*$", header, re.MULTILINE
        )
        if name_match is None:
            errors.append(f"{skill_file.relative_to(ROOT)}: missing name")
            continue
        name = name_match.group(1).strip()
        if len(name) > 64 or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None:
            errors.append(
                f"{skill_file.relative_to(ROOT)}: name must be 1-64 kebab-case characters"
            )
        if name != skill_file.parent.name:
            errors.append(
                f"{skill_file.relative_to(ROOT)}: name {name!r} does not match directory"
            )
        if name in seen:
            errors.append(f"duplicate skill name: {name}")
        seen.add(name)
        if desc_match is None:
            errors.append(f"{skill_file.relative_to(ROOT)}: missing description")
        else:
            description = desc_match.group("description").strip()
            if (
                len(description) >= 2
                and description[0] == description[-1]
                and description[0] in "\"'"
            ):
                description = description[1:-1].strip()
            if not description or len(description) > 1024:
                errors.append(
                    f"{skill_file.relative_to(ROOT)}: description must be 1-1024 characters"
                )
        compatibility_paths = (
            ".claude/skills",
            ".pi/skills",
            ".dsh/skills",
            "CLAUDE.md",
        )
        if any(path in text for path in compatibility_paths):
            errors.append(
                f"{skill_file.relative_to(ROOT)}: references a harness compatibility path"
            )
        for human_path in ("docs/en/", "CONTRIBUTING.md"):
            if human_path in text:
                errors.append(
                    f"{skill_file.relative_to(ROOT)}: depends on human documentation {human_path!r}"
                )
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in target or target.startswith(("#", "/")):
                continue
            resolved = skill_file.parent / target.split("#", 1)[0]
            if not resolved.exists():
                errors.append(
                    f"{skill_file.relative_to(ROOT)}: missing linked resource {target!r}"
                )
        for obsolete in ("tool/", "modern-tsf", "MODEL_NAME_MAP", "registry.py", "schema.py"):
            if obsolete in text:
                errors.append(
                    f"{skill_file.relative_to(ROOT)}: references obsolete interface {obsolete!r}"
                )

    if seen != EXPECTED_SKILLS:
        missing = sorted(EXPECTED_SKILLS - seen)
        unexpected = sorted(seen - EXPECTED_SKILLS)
        if missing:
            errors.append(f"missing canonical skills: {', '.join(missing)}")
        if unexpected:
            errors.append(f"unexpected or obsolete skills: {', '.join(unexpected)}")
    from moderntsf.tsf_core.agent_tasks import audit_tasks

    errors.extend(audit_tasks())
    return errors


def main() -> int:
    errors = audit_agent_assets()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    count = len(list(SKILLS.glob("*/SKILL.md")))
    print(f"Cross-harness Agent-first assets OK: {count} skills")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

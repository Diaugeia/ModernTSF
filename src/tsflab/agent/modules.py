"""Module chain (Data -> Models -> Experiments -> Release, AutoResearch) and project records.

A project created by ``tsf init`` records its chosen modules under
``[tool.tsflab]`` in its ``pyproject.toml``; ``tsf agent`` and ``tsf catalog``
read that record. Skills and task templates for the chosen modules live in the
project's ``.agents/`` directory and are refreshed from the installed package
with ``tsf agent sync``.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

# module -> (skills, tasks, pip extras, one-line purpose, entry commands)
MODULES: dict[str, dict[str, object]] = {
    "data": {
        "skills": ["add-dataset", "inspect-dataset"],
        "tasks": [],
        "extras": ["data"],
        "purpose": "register, prepare, profile, and publish datasets",
        "commands": ["tsf data inspect", "tsf data analyze", "tsf data prepare", "tsf data add"],
    },
    "models": {
        "skills": ["discover-papers", "add-model", "integrate-foundation-model", "curate-components"],
        "tasks": ["intake"],
        "extras": ["models"],
        "purpose": "find papers, implement or adapt models, curate reusable components",
        "commands": ["tsf model scaffold", "tsf model add", "tsf model verify", "tsf model compose"],
    },
    "experiments": {
        "skills": ["setup-environment", "run-experiment", "diagnose-experiment",
                   "reproduce-paper-results", "analyze-results"],
        "tasks": ["experiment"],
        "extras": ["experiments"],
        "purpose": "design, run, diagnose, and analyze budgeted experiments",
        "commands": ["tsf env", "tsf run", "tsf result aggregate", "tsf result rank"],
    },
    "release": {
        "skills": ["submit-results", "forecast-realtime-round", "publish-weights"],
        "tasks": [],
        "extras": ["hub", "realtime"],
        "purpose": "submit results, forecast real-time rounds, publish weights",
        "commands": ["tsf result submit", "tsf result hub", "tsf realtime update"],
    },
    "autoresearch": {
        "skills": ["run-autoresearch"],
        "tasks": ["autoresearch"],
        "extras": ["autoresearch"],
        "purpose": "budgeted research loops that consume the other modules' context",
        "commands": ["tsf research start", "tsf research iteration", "tsf agent task start autoresearch"],
    },
}
CHAIN = tuple(MODULES)


def parse_modules(text: str | None) -> list[str]:
    """Parse ``a,b,c`` (default: every module) and reject unknown names."""
    if text is None or text.strip() in {"", "all"}:
        return list(CHAIN)
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in MODULES]
    if unknown:
        raise ValueError(f"unknown module(s): {', '.join(unknown)}; choose from {', '.join(CHAIN)}")
    return [name for name in CHAIN if name in names]


def module_tasks(modules: list[str]) -> list[str]:
    return [task for name in modules for task in MODULES[name]["tasks"]]  # type: ignore[union-attr]


def module_skills(modules: list[str], task_skills: dict[str, list[str]] | None = None) -> list[str]:
    """Skills of the modules plus any skill a selected task template requires."""
    skills: list[str] = []
    for name in modules:
        skills.extend(MODULES[name]["skills"])  # type: ignore[arg-type]
    for task in module_tasks(modules):
        skills.extend((task_skills or {}).get(task, []))
    return list(dict.fromkeys(skills))


def find_project(start: Path | None = None) -> tuple[Path, list[str]] | None:
    """Return ``(project_dir, modules)`` for the nearest ``[tool.tsflab]`` pyproject."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        pyproject = directory / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            table = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("tool", {}).get("tsflab")
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if isinstance(table, dict) and isinstance(table.get("modules"), list):
            return directory, [m for m in table["modules"] if m in MODULES]
    return None


def project_modules(start: Path | None = None) -> list[str] | None:
    found = find_project(start)
    return found[1] if found else None

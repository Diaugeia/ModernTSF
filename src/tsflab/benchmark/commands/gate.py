"""tsf repo check — the single definition of "mergeable".

    tsf repo check [--scope full|changed] [--base REF] [--fail-fast] [--json]

``full`` runs every check. ``changed`` runs the same checks plus the targeted
smoke run for the models affected by the diff against ``--base`` (default:
``origin/$GITHUB_BASE_REF`` in CI, else the first of origin/dev, dev,
origin/main, main that exists). The full ``tsf smoke --all`` sweep stays a
nightly/manual job. Fast steps run first; every step runs unless
``--fail-fast``; the exit code is 0 only when all steps pass.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tsflab.tsf_core.paths import repository_root

PY = sys.executable
CLI = [PY, "-m", "tsflab.benchmark.cli"]

# Diff paths whose change warrants a targeted smoke run (smoke-affected.yml).
SMOKE_TRIGGERS = (
    "src/tsflab/models/",
    "src/tsflab/benchmark/",
    "src/tsflab/data/",
    "src/tsflab/tsf_core/",
    "tests/",
    "configs/runs/smoke_",
)
# Changes that touch every model at once map to the representative set.
SMOKE_BROAD = (
    "src/tsflab/benchmark/",
    "src/tsflab/models/_components/",
    "src/tsflab/data/",
    "src/tsflab/tsf_core/",
    "tests/",
)
SMOKE_FALLBACK = (
    "configs/runs/smoke_crib.toml",
    "configs/runs/smoke_quantile_dlinear.toml",
    "configs/runs/smoke_gaussian_mlp.toml",
)


@dataclass(frozen=True)
class Step:
    name: str
    argv: list[str]
    cwd: str = "."
    scopes: tuple[str, ...] = ("full", "changed")
    setup: list[list[str]] = field(default_factory=list)


def affected_smoke_configs(changed: list[str], root: Path) -> list[str]:
    """Map changed files to smoke configs; empty when nothing needs smoking."""
    if not any(path.startswith(SMOKE_TRIGGERS) for path in changed):
        return []
    if any(path.startswith(SMOKE_BROAD) for path in changed):
        return list(SMOKE_FALLBACK)
    configs: set[str] = set()
    model_changed = False
    for path in changed:
        parts = path.split("/")
        if path.startswith("src/tsflab/models/") and len(parts) > 3:
            model_changed = True
            for found in sorted((root / "configs/runs").glob(f"smoke_{parts[3]}*.toml")):
                configs.add(found.relative_to(root).as_posix())
        if path.startswith("configs/runs/smoke_") and path.endswith(".toml"):
            if (root / path).is_file():
                configs.add(path)
    if not configs and model_changed:
        return list(SMOKE_FALLBACK)
    return sorted(configs)


def select_steps(scope: str, changed: list[str], root: Path) -> list[Step]:
    """Return the ordered steps for ``scope`` (pure; runs nothing)."""
    steps = [
        Step("schema-export", [PY, "-m", "tsflab.tsf_core", "--check",
                               "--out-dir", "src/tsflab/tsf_core/schema"]),
        Step("agent-assets", [PY, "-m", "tsflab.tsf_core.agent_assets"]),
        Step("agent-tasks", CLI + ["agent", "task", "validate"]),
        Step("model-cards", CLI + ["model", "audit", "--summary"]),
        Step("verification-stale", CLI + ["verify", "stale"]),
        Step("dataset-cards", CLI + ["dataset", "audit"]),
        Step("component-cards", CLI + ["component", "audit"]),
        Step("web-submissions", [PY, "pipeline/validate.py"], cwd="apps/web"),
        Step("repo-audit", CLI + ["repo", "audit"]),
    ]
    if scope == "changed":
        configs = affected_smoke_configs(changed, root)
        if configs:
            steps.append(Step(
                "smoke-affected",
                CLI + ["smoke", "--config", *configs],
                setup=[[PY, "scripts/make_smoke_data.py"]],
            ))
    steps.append(Step("pytest", [PY, "-m", "pytest", "-q", "tests"]))
    return steps


def changed_files(root: Path, base: str | None) -> list[str]:
    """Files changed on this branch relative to ``base`` (merge-base diff)."""
    candidates = [base] if base else []
    if not base:
        env = os.environ.get("GITHUB_BASE_REF")
        if env:
            candidates.append(f"origin/{env}")
        candidates += ["origin/dev", "dev", "origin/main", "main"]
    for ref in candidates:
        probe = subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref],
                               cwd=root, capture_output=True)
        if probe.returncode == 0:
            diff = subprocess.run(["git", "diff", "--name-only", f"{ref}...HEAD"],
                                  cwd=root, capture_output=True, text=True)
            if diff.returncode == 0:
                tracked = diff.stdout.split()
                dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root,
                                       capture_output=True, text=True).stdout
                return sorted(set(tracked) | {line[3:] for line in dirty.splitlines()})
    raise RuntimeError("cannot resolve a base ref for --scope changed; pass --base REF")


def _run(step: Step, root: Path, quiet: bool) -> dict:
    start = time.monotonic()
    code = 0
    output = ""
    for command in [*step.setup, step.argv]:
        proc = subprocess.run(command, cwd=root / step.cwd, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONUNBUFFERED": "1"})
        output += proc.stdout
        code = proc.returncode
        if code:
            break
    if not quiet and code:
        print(output, file=sys.stderr)
    return {"step": step.name, "ok": code == 0, "code": code,
            "seconds": round(time.monotonic() - start, 1),
            "tail": "\n".join(output.strip().splitlines()[-15:])}


def check_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="tsf repo check", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scope", choices=["full", "changed"], default="full")
    parser.add_argument("--base", help="base ref for --scope changed")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--list", action="store_true", help="print the steps and exit")
    args = parser.parse_args(argv)
    root = repository_root()
    try:
        changed = changed_files(root, args.base) if args.scope == "changed" else []
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    steps = select_steps(args.scope, changed, root)
    if args.list:
        for step in steps:
            print(f"{step.name}: {' '.join(step.argv[1:])}")
        return 0
    results = []
    for step in steps:
        if not args.json:
            print(f"[check] {step.name} ...", flush=True)
        result = _run(step, root, quiet=args.json)
        results.append(result)
        if not args.json:
            print(f"[check] {step.name}: {'PASS' if result['ok'] else 'FAIL'} "
                  f"({result['seconds']}s)", flush=True)
        if args.fail_fast and not result["ok"]:
            break
    failed = [r["step"] for r in results if not r["ok"]]
    if args.json:
        print(json.dumps({"scope": args.scope, "passed": not failed, "failed": failed,
                          "steps": results}, indent=2))
    else:
        print("\n== tsf repo check summary ==")
        for r in results:
            print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['step']}  ({r['seconds']}s)")
        print("MERGEABLE" if not failed else f"NOT MERGEABLE: {', '.join(failed)}")
    return 1 if failed else 0

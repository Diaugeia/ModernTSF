#!/usr/bin/env python3
"""Public Agent CLI for TSFLab.

The CLI is intentionally a thin router. Command behavior lives in focused
modules so the public surface stays stable while model, data, execution, and
repository concerns evolve independently.

Usage:
    tsf <command> [args...]

Catalog and resource operations:
    model            add, list, show, or audit a model specification
    component        list, match, or show a reusable implementation component
    dataset          add, prepare, inspect, analyze, or plot a dataset
    catalog          search models, components, and datasets (ranked L0 lines)
    result           aggregate, rank, plot, or report results
    repo             check (mergeable gate), audit, diagnose, or regenerate cards
    verify           run or inspect unified model verification
    agent            list, inspect, validate, render, or start bounded Agent tasks

Execution:
    smoke            run smoke configurations concurrently
    run              run experiment configurations concurrently
    inspect          preview resolved configuration expansion
    env              audit dependencies, accelerators, data, and output capacity
    interface        discover public workflows and execution policy schema
    queue            optionally queue prepared sweeps with priorities
    slurm            explicitly submit, inspect, or cancel a cluster sweep
    storage          inspect capacity and preview managed checkpoint cleanup
    usage            reserve and settle external token/USD spending

Project and publishing:
    init             scaffold a standalone project on the installed package
    hub              pack, push, list, or pull weights on the Hugging Face Hub
    realtime         rolling real-time tracks: releases, rounds, forecasts, scores

Records and integration:
    research         manage lightweight research rounds
    submit           package a run into a Submission Report
    schema-export    export TSF-Core JSON Schema
    leaderboard-build  recompute a leaderboard from submissions

Progressive disclosure: search returns L0 lines; ``model|component|dataset show
<name> --depth {0,1,2,3}`` opens L0 line, L1 interface/constraints, L2 full card,
or L3 paths to open.

Run ``tsf <command> --help`` for command-specific options.
"""

from __future__ import annotations

import sys

from tsflab.benchmark.command_runtime import passthrough


def schema_export_command(rest: list[str]) -> int:
    """Export the lightweight TSF-Core contract models to JSON Schema."""
    from tsflab.tsf_core.export import main as schema_main

    return schema_main(rest)


def main(argv: list[str] | None = None) -> int:
    """Dispatch one public command without importing unrelated heavy modules."""
    argv = sys.argv[1:] if argv is None else argv
    if argv[:2] == ["--format", "json"]:
        from tsflab.benchmark.commands.envelope import envelope
        return envelope(argv[2:])
    if argv and argv[0] in {"queue", "usage", "storage", "slurm"}:
        from tsflab.benchmark.commands.operations import operations_command
        return operations_command(argv)
    if argv and argv[0] in {"env", "interface"}:
        from tsflab.benchmark.commands.infrastructure import infrastructure_command
        return infrastructure_command(argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(__doc__)
        return 0

    command, rest = argv[0], argv[1:]
    if command in {"smoke", "run"}:
        from tsflab.benchmark.commands.execution import run_command, smoke_command

        return smoke_command(rest) if command == "smoke" else run_command(rest)
    if command in {"model", "component"}:
        from tsflab.benchmark.commands.catalog_resources import (
            component_command,
            model_command,
        )

        handlers = {
            "model": model_command,
            "component": component_command,
        }
        return handlers[command](rest)
    if command == "catalog":
        from tsflab.benchmark.commands.catalog_resources import catalog_command

        return catalog_command(rest)
    if command in {"dataset", "result"}:
        from tsflab.benchmark.commands.data_results import dataset_command, result_command

        return dataset_command(rest) if command == "dataset" else result_command(rest)
    if command == "repo":
        from tsflab.benchmark.commands.repository import repository_command

        return repository_command(rest)
    if command == "verify":
        from tsflab.benchmark.commands.verification import verification_command

        return verification_command(rest)
    if command == "agent":
        from tsflab.benchmark.commands.agent_tasks import agent_command

        return agent_command(rest)
    if command == "research":
        from tsflab.benchmark.commands.research import research_command

        return research_command(rest)
    if command == "schema-export":
        return schema_export_command(rest)
    if command == "init":
        from tsflab.scaffold import main as init_main

        return init_main(rest)
    if command == "realtime":
        from tsflab.realtime.cli import main as realtime_main

        return realtime_main(rest)
    if command == "hub":
        from tsflab.benchmark.commands.hub import hub_command

        return hub_command(rest)

    passthrough_commands = {
        "inspect": "inspect_config.py",
        "submit": "submit.py",
        "leaderboard-build": "leaderboard_build.py",
    }
    script = passthrough_commands.get(command)
    if script is not None:
        return passthrough(script, rest)

    print(f"unknown command: {command!r}\n", file=sys.stderr)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

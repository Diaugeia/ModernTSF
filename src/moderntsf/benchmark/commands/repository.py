"""Repository consistency and executable model-contract commands."""

from __future__ import annotations

import argparse
import sys

from moderntsf.benchmark.command_runtime import passthrough


def repository_command(args: list[str]) -> int:
    """Audit static repository contracts and optionally execute all model contracts."""
    if not args or args[0] in {"-h", "--help", "help"}:
        print("usage: tsf repo {audit,doctor} [--forward | --backward | --strict]")
        return 0
    action, rest = args[0], args[1:]
    if action not in {"audit", "doctor"}:
        print(
            "usage: tsf repo {audit,doctor} [--forward | --backward | --strict]",
            file=sys.stderr,
        )
        return 2
    if action == "audit" and rest:
        print("tsf repo audit takes no arguments", file=sys.stderr)
        return 2

    forward = False
    backward = False
    if action == "doctor":
        parser = argparse.ArgumentParser(prog="tsf repo doctor")
        parser.add_argument("--forward", action="store_true", help="run every tensor contract")
        parser.add_argument(
            "--backward",
            action="store_true",
            help="also verify finite gradients (implies --forward)",
        )
        parser.add_argument(
            "--strict",
            action="store_true",
            help=(
                "also verify gradients, batch-size-one execution, and an exact "
                "state-dict/output round trip"
            ),
        )
        parser.add_argument(
            "--models",
            nargs="+",
            metavar="NAME",
            help="check only these models (default: the complete catalog)",
        )
        parsed = parser.parse_args(rest)
        forward = parsed.forward or parsed.backward or parsed.strict
        backward = parsed.backward or parsed.strict
        strict = parsed.strict
        selected_models = parsed.models
    else:
        selected_models = None
        strict = False

    from moderntsf.tsf_core.agent_assets import main as audit_agent_assets
    from moderntsf.benchmark.resource_cards import audit_resource_cards
    from moderntsf.tsf_core.paths import repository_root

    def audit_cards() -> int:
        errors = audit_resource_cards(repository_root())
        for error in errors:
            print(f"ERROR: {error}")
        if not errors:
            print("Resource cards OK")
        return 1 if errors else 0

    checks = [
        ("agent-assets", audit_agent_assets),
        ("components", lambda: __import__("moderntsf.benchmark.catalog.component_audit", fromlist=["main"]).main()),
        ("resource-cards", audit_cards),
        ("model-catalog", lambda: passthrough("check_registry.py", [])),
        ("documentation", lambda: passthrough("check_docs.py", [])),
    ]
    results = []
    for name, check in checks:
        code = check()
        results.append(code)
        print(f"{'PASS' if code == 0 else 'FAIL'} {name}")

    if action == "doctor":
        from moderntsf.benchmark.model_contracts import audit_model_contracts
        from moderntsf.benchmark.registry.models import MODEL_CATALOG

        names = selected_models or MODEL_CATALOG.names()
        failed = audit_model_contracts(
            names=names, forward=forward, backward=backward, strict=strict
        )
        for failure in failed:
            print(f"FAIL {failure.stage} {failure.model}: {failure.error}")
        action_name = (
            "strict-checked"
            if strict
            else "backward-checked"
            if backward
            else "forward-checked"
            if forward
            else "constructed"
        )
        print(
            f"{action_name.capitalize()} "
            f"{len(names) - len(failed)}/{len(names)} models"
        )
        results.append(1 if failed else 0)
    return 1 if any(results) else 0

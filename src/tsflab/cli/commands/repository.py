"""Repository consistency and executable model-contract commands."""

from __future__ import annotations

import sys

from tsflab.cli.runtime import passthrough


def regenerate_cards(args: list[str]) -> int:
    """Rewrite every generated card and index from the catalogs (checkout only)."""
    if args:
        print("tsf repo cards takes no arguments", file=sys.stderr)
        return 2
    from tsflab.catalog.cards.resources import write_resource_cards
    from tsflab.core.paths import require_checkout

    try:
        root = require_checkout("tsf repo cards")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    from tsflab.catalog.cards.models import update_model_card

    models = sorted(
        card for card in (root / "src" / "tsflab" / "models").glob("*/README.md")
        if not card.parent.name.startswith("_")
    )
    changed = sum(update_model_card(card) for card in models)
    print(f"Regenerated model card bodies: {changed} of {len(models)} changed")
    count = write_resource_cards(root)
    print(f"Generated {count} component and dataset cards")
    return passthrough("check_docs.py", ["--write"])


def run_audit() -> int:
    """Run the static repository audits in-process; return a nonzero code on failure."""
    from tsflab.agent.assets import main as audit_agent_assets
    from tsflab.catalog.cards.resources import audit_resource_cards
    from tsflab.core.paths import repository_root

    def audit_cards() -> int:
        errors = audit_resource_cards(repository_root())
        for error in errors:
            print(f"ERROR: {error}")
        if not errors:
            print("Resource cards OK")
        return 1 if errors else 0

    checks = [
        ("agent-assets", audit_agent_assets),
        ("components", lambda: __import__("tsflab.catalog.component_audit", fromlist=["main"]).main()),
        ("resource-cards", audit_cards),
        ("model-catalog", lambda: passthrough("check_registry.py", [])),
        ("documentation", lambda: passthrough("check_docs.py", [])),
    ]
    results = []
    for name, check in checks:
        code = check()
        results.append(code)
        print(f"{'PASS' if code == 0 else 'FAIL'} {name}")
    return 1 if any(results) else 0


def run_contracts(level: str, models: list[str] | None) -> int:
    """Execute model tensor contracts at ``level`` (construct, forward, backward, strict)."""
    from tsflab.catalog.model_contracts import audit_model_contracts
    from tsflab.catalog.registry.models import MODEL_CATALOG

    strict = level == "strict"
    backward = level in {"backward", "strict"}
    forward = backward or level == "forward"
    names = models or MODEL_CATALOG.names()
    failed = audit_model_contracts(names=names, forward=forward, backward=backward, strict=strict)
    for failure in failed:
        print(f"FAIL {failure.stage} {failure.model}: {failure.error}")
    label = {"strict": "strict-checked", "backward": "backward-checked",
             "forward": "forward-checked"}.get(level, "constructed")
    print(f"{label.capitalize()} {len(names) - len(failed)}/{len(names)} models")
    return 1 if failed else 0


def repository_command(args: list[str]) -> int:
    """Check repository contracts, regenerate cards, or export the TSF-Core schema."""
    usage = "usage: tsf repo {check,cards,schema} [args...]"
    if not args or args[0] in {"-h", "--help", "help"}:
        print(usage)
        print("  check   the single mergeable gate: [--scope full|changed] [--only STEP...] [--json]")
        print("          --audit runs the static audits only; --contracts construct|forward|backward|strict")
        print("          [--models NAME...] executes model tensor contracts")
        print("  cards   regenerate component/dataset cards and the model documentation index")
        print("  schema  export TSF-Core JSON Schema [--check] [--out-dir DIR]")
        return 0
    action, rest = args[0], args[1:]
    if action == "check":
        from tsflab.cli.commands.gate import check_command

        return check_command(rest)
    if action == "cards":
        return regenerate_cards(rest)
    if action == "schema":
        from tsflab.core.export import main as schema_main

        return schema_main(rest)
    print(usage, file=sys.stderr)
    return 2

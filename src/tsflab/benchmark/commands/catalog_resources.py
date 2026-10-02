"""Public inspection commands for the flat model and component catalogs."""

from __future__ import annotations

import json
import sys
from collections import Counter

from tsflab.benchmark.command_runtime import ROOT, passthrough


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _model_audit_record(
    fields: dict[str, object],
) -> dict[str, object]:
    """Return one machine-readable model-card and verification gate result."""
    paper = dict(fields.get("paper", {}))
    codebase_value = fields.get("codebase")
    codebase = dict(codebase_value) if isinstance(codebase_value, dict) else {}
    missing_source = [
        field
        for field in ("url", "revision", "license")
        if not codebase.get(field)
    ]
    blockers = []
    if not paper.get("title"):
        blockers.append("paper.title")
    if codebase:
        blockers.extend(f"codebase.{field}" for field in missing_source)
    from tsflab.benchmark.verification import evidence_state

    state = evidence_state(ROOT, str(fields["name"]), fields)
    verification_status: dict[str, object] = {
        "status": state.status,
        "current": state.current,
        "evidence": state.evidence,
    }
    if state.detail:
        verification_status["detail"] = state.detail
    if state.status != "passed" or not state.current:
        blockers.append("verification.failed")
    return {
        "name": str(fields["name"]),
        "passed": not blockers,
        "blockers": blockers,
        "paper": {
            "title": paper.get("title", ""),
            "venue": paper.get("venue", ""),
            "year": paper.get("year"),
            "url": paper.get("url", ""),
        },
        "codebase": {
            "url": codebase.get("url", ""),
            "revision": codebase.get("revision", ""),
            "license": codebase.get("license", ""),
            "missing": missing_source,
        },
        "smoke_config": fields.get("smoke_config"),
        "components": list(fields.get("components", ())),
        "verification": verification_status,
    }


def model_command(args: list[str]) -> int:
    """Add, list, describe, or audit named model and method specifications."""
    if not args or args[0] in {"-h", "--help", "help"}:
        print(
            "usage: tsf model {scaffold,add,list,show,search,artifacts,audit} [args...]\n"
            "       tsf model list [--details | --json]\n"
            "       tsf model search <terms...> [--capability C] [--limit N] [--json]   (L0 lines)\n"
            "       tsf model show <name> [--depth {0,1,2,3}] [--json]"
        )
        return 0
    action, rest = args[0], args[1:]
    if action == "scaffold":
        return passthrough("new_model.py", rest)
    if action == "add":
        return passthrough("finalize_model.py", rest)

    if action == "list":
        if any(arg not in {"--details", "--json"} for arg in rest) or len(rest) > 1:
            print("usage: tsf model list [--details | --json]", file=sys.stderr)
            return 2
        from tsflab.benchmark.cards.metadata import model_records

        fields_by_name = model_records(ROOT)
        if not rest:
            print("\n".join(sorted(str(fields["name"]) for fields in fields_by_name)))
            return 0
        from tsflab.benchmark.cards.descriptions import read_model_card_description

        records = []
        for fields in fields_by_name:
            model_card = str(fields["model_card"])
            capabilities = set(fields.get("capabilities", ()))
            task_modes = {
                public
                for capability, public in {
                    "time-series": "time_series",
                    "spatiotemporal": "spatiotemporal",
                    "covariate": "covariate",
                }.items()
                if capability in capabilities
            }
            records.append(
                {
                    "name": str(fields["name"]),
                    "summary": read_model_card_description(ROOT / model_card).summary,
                    "capabilities": sorted(capabilities),
                    "task_modes": sorted(task_modes),
                }
            )
        if rest == ["--json"]:
            _print(records)
        else:
            for record in records:
                print(f"{record['name']}\n  {record['summary']}")
        return 0
    if action == "show":
        from tsflab.benchmark.cards.metadata import model_records
        from tsflab.benchmark.cards.show import existing, parse_show, show_card
        from tsflab.benchmark.registry.models import MODEL_CATALOG

        parsed = parse_show("tsf model show", "public model name", rest)
        try:
            spec = MODEL_CATALOG.get(parsed.name)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        fields = next(
            record for record in model_records(ROOT) if record["name"] == spec.name
        )
        paper = dict(fields["paper"])
        codebase = fields["codebase"]
        card_path = ROOT / str(fields["model_card"])
        fields["card_text"] = card_path.read_text(encoding="utf-8")
        audit = _model_audit_record(fields)
        legacy = {
            "name": spec.name,
            "module": spec.module,
            "summary": fields["summary"],
            "parameters": spec.params_schema.model_json_schema(),
            "paper": {
                "title": paper["title"],
                "venue": paper["venue"],
                "year": paper["year"],
                "url": paper["url"],
            },
            "codebase": codebase,
            "config": spec.config_path,
            "model_card": spec.model_card,
            "smoke_config": spec.smoke_config,
            "capabilities": sorted(spec.capabilities),
            "components": list(spec.components),
            "output_type": spec.output_type,
            "task_modes": sorted(spec.task_modes),
            "artifacts": [
                {
                    "name": artifact.name,
                    "revision": artifact.revision,
                    "filename": artifact.filename,
                    "required": artifact.required,
                }
                for artifact in spec.artifacts
            ],
            "verification": audit["verification"],
            "blockers": audit["blockers"],
        }
        facts = {
            "config": spec.config_path,
            "smoke_config": spec.smoke_config or "(none)",
            "task_modes": sorted(spec.task_modes),
            "capabilities": sorted(spec.capabilities),
            "components": list(spec.components),
            "output_type": spec.output_type,
            "verification": audit["verification"]["status"],
            "blockers": audit["blockers"],
        }
        package = card_path.parent.relative_to(ROOT).as_posix()
        paths = existing(
            ROOT,
            card_path.relative_to(ROOT).as_posix(),
            f"{package}/model.py",
            str(fields["spec_file"]),
            spec.config_path,
            spec.smoke_config or "",
            f"verification/evidence/{spec.name}.json",
            *(
                f"src/tsflab/models/_components/{name}/README.md"
                for name in spec.components
            ),
        )
        return show_card(ROOT, card_path, parsed, facts=facts, paths=paths, legacy=legacy)
    if action == "artifacts":
        import argparse
        from pathlib import Path

        from tsflab.benchmark.model_artifacts import artifact_status, fetch_artifact
        from tsflab.benchmark.registry.models import MODEL_CATALOG

        parser = argparse.ArgumentParser(
            prog="tsf model artifacts",
            description="Inspect or explicitly fetch checksum-pinned model artifacts.",
        )
        parser.add_argument("name", help="public model name")
        parser.add_argument("--fetch", metavar="ARTIFACT", help="artifact name to download")
        parser.add_argument("--cache-dir", type=Path)
        parser.add_argument("--json", action="store_true")
        parsed = parser.parse_args(rest)
        try:
            spec = MODEL_CATALOG.get(parsed.name)
        except KeyError as exc:
            parser.error(str(exc))
        if parsed.fetch:
            selected = next(
                (item for item in spec.artifacts if item.name == parsed.fetch), None
            )
            if selected is None:
                parser.error(f"model {spec.name!r} has no artifact {parsed.fetch!r}")
            fetch_artifact(spec, selected, parsed.cache_dir)
        records = artifact_status(spec, parsed.cache_dir)
        if parsed.json:
            _print(records)
        elif not records:
            print(f"{spec.name} declares no external runtime artifacts")
        else:
            for record in records:
                state = "verified" if record["verified"] else "missing-or-invalid"
                print(f"{record['name']}\t{state}\t{record['path']}")
        return 0
    if action == "search":
        from tsflab.benchmark.cards.search import search_command

        return search_command(ROOT, rest, prog="tsf model search", kind="model")
    if action == "audit":
        import argparse
        from tsflab.benchmark.cards.metadata import model_records

        parser = argparse.ArgumentParser(
            prog="tsf model audit",
            description="Audit model cards and executable verification evidence.",
        )
        parser.add_argument("names", nargs="*", help="model names; default: all")
        output = parser.add_mutually_exclusive_group()
        output.add_argument("--json", action="store_true", help="emit per-model JSON")
        output.add_argument("--summary", action="store_true", help="emit aggregate JSON")
        parsed = parser.parse_args(rest)
        declared = {str(fields["name"]): fields for fields in model_records(ROOT)}
        names = parsed.names or sorted(declared)
        unknown = [name for name in names if name not in declared]
        if unknown:
            print(f"Unknown model(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        for name in names:
            card = ROOT / str(declared[name]["model_card"])
            declared[name]["card_text"] = card.read_text(encoding="utf-8")
        records = [
            _model_audit_record(declared[name])
            for name in names
        ]
        failures = [record for record in records if not record["passed"]]
        if parsed.summary:
            blockers = Counter(
                blocker for record in failures for blocker in record["blockers"]
            )
            _print(
                {
                    "models": len(records),
                    "passed": len(records) - len(failures),
                    "failed": len(failures),
                    "blockers": dict(sorted(blockers.items())),
                    "verification": dict(
                        sorted(
                            Counter(
                                str(r["verification"]["status"])
                                for r in records
                            ).items()
                        )
                    ),
                    "complete_codebase": sum(
                        bool(r["codebase"]["url"]) and not r["codebase"]["missing"]
                        for r in records
                    ),
                    "with_smoke_config": sum(bool(r["smoke_config"]) for r in records),
                }
            )
        elif parsed.json:
            _print(records)
        else:
            for record in failures:
                print(f"FAIL {record['name']}: {', '.join(record['blockers'])}")
            print(
                f"{len(records) - len(failures)}/{len(records)} model audits passed"
            )
        return 1 if failures else 0
    print(f"unknown model action: {action!r}", file=sys.stderr)
    return 2

def component_command(args: list[str]) -> int:
    """List, match, or describe shared components and their consumers."""
    from tsflab.benchmark.catalog.component_audit import components_used_by
    from tsflab.benchmark.catalog.components import COMPONENT_CATALOG

    if not args or args[0] in {"-h", "--help", "help"}:
        print(
            "usage: tsf component {list,show,search,match,compose,audit} [args...]\n"
            "       tsf component list [--json]                (L0 lines)\n"
            "       tsf component compose <spec.toml> [--json]   (dry run)\n"
            "       tsf component search <terms...> [--limit N] [--json]   (L0 lines; match is an alias)\n"
            "       tsf component show <name> [--depth {0,1,2,3}] [--json]"
        )
        return 0
    action, rest = args[0], args[1:]
    if action == "compose":
        from tsflab.benchmark.catalog.composition import compose_command

        return compose_command(rest, ROOT)
    if action == "audit":
        if rest:
            print("tsf component audit takes no arguments", file=sys.stderr)
            return 2
        from tsflab.benchmark.cards.resources import audit_resource_cards
        from tsflab.benchmark.catalog.component_audit import audit_components
        from tsflab.tsf_core.paths import repository_root

        root = repository_root()
        failures = audit_components()
        failures.extend(
            error for error in audit_resource_cards(root) if "models/_components" in error
        )
        for failure in failures:
            print(f"ERROR: {failure}")
        total = len(COMPONENT_CATALOG.names())
        print(f"Component catalog/cards: {'PASS' if not failures else 'FAIL'} ({total} components)")
        return 1 if failures else 0
    if action == "list" and (not rest or rest == ["--json"]):
        from tsflab.benchmark.cards.depth import card_l0, l0_line, read_card
        from tsflab.benchmark.cards.components import component_card_path

        records = []
        for spec in COMPONENT_CATALOG.specs():
            record = card_l0(read_card(component_card_path(ROOT, spec.name)))
            record.update(
                module=spec.module,
                card=f"src/tsflab/models/_components/{spec.name}/README.md",
            )
            records.append(record)
        if rest == ["--json"]:
            _print(records)
        else:
            for record in records:
                print(l0_line(record))
        return 0
    if action == "show":
        from tsflab.benchmark.cards.show import existing, parse_show, show_card
        from tsflab.benchmark.cards.components import component_card_path

        parsed = parse_show("tsf component show", "component name", rest)
        try:
            spec = COMPONENT_CATALOG.get(parsed.name)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        consumers = []
        for package in sorted((ROOT / "src" / "tsflab" / "models").iterdir()):
            if package.is_dir() and spec.name in components_used_by(package):
                consumers.append(package.name)
        card = f"src/tsflab/models/_components/{spec.name}/README.md"
        legacy = {
            "name": spec.name,
            "module": spec.module,
            "contract": spec.contract,
            "public_symbols": list(spec.public_symbols),
            "keywords": list(spec.keywords),
            "consumers": consumers,
            "card": card,
        }
        shown = ", ".join(consumers[:8]) + (", ..." if len(consumers) > 8 else "")
        facts = {
            "public_symbols": list(spec.public_symbols),
            "consumers": f"{len(consumers)} models: {shown}" if consumers else "none",
        }
        tests = sorted(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "tests").glob("*.py")
            if spec.module in path.read_text(encoding="utf-8")
        )
        paths = existing(
            ROOT,
            card,
            f"src/tsflab/models/_components/{spec.name}/__init__.py",
            *tests[:6],
            *(f"src/tsflab/models/{name}/model.py" for name in consumers[:3]),
        )
        return show_card(
            ROOT, component_card_path(ROOT, spec.name), parsed, facts=facts, paths=paths, legacy=legacy
        )
    if action in {"match", "search"}:
        from tsflab.benchmark.cards.search import search_command

        def augment(record: dict[str, object]) -> dict[str, object]:
            spec = COMPONENT_CATALOG.get(str(record["name"]))
            return {
                **record,
                "contract": spec.contract,
                "module": spec.module,
                "review_required": True,
            }

        code = search_command(
            ROOT, rest, prog=f"tsf component {action}", kind="component", augment=augment
        )
        if code == 0 and "--json" not in rest:
            print(
                "Candidate retrieval only; open one with `tsf component show <name> "
                "--depth 1` and review shapes and semantics before reuse.",
                file=sys.stderr,
            )
        return code
    print("usage: tsf component {list,show,search,match,compose,audit} [args...]", file=sys.stderr)
    return 2


def catalog_command(args: list[str]) -> int:
    """Search all catalogs at once, returning ranked L0 lines."""
    if not args or args[0] in {"-h", "--help", "help"} or args[0] != "search":
        print(
            "usage: tsf catalog search <terms...> [--kind model|component|dataset] "
            "[--limit N] [--json]\n"
            "Each result is one L0 line: name, kind, summary, tags. Open one with\n"
            "`tsf <kind> show <name> --depth 1|2|3`."
        )
        return 0 if not args or args[0] in {"-h", "--help", "help"} else 2
    from tsflab.benchmark.cards.search import search_command

    return search_command(ROOT, args[1:], prog="tsf catalog search")

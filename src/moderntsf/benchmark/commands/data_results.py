"""Dataset and result resource command routing behind the public CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from moderntsf.benchmark.command_runtime import ROOT, passthrough


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _dataset_record_payload(record: object, facts: dict[str, object] | None = None) -> dict[str, object]:
    from dataclasses import asdict

    payload = asdict(record)
    payload["card"] = f"catalog/datasets/{record.name}/README.md"
    if facts is not None:
        payload["facts"] = facts  # curated card front matter (level 0/1)
    return payload


def dataset_command(args: list[str]) -> int:
    """Route dataset scaffolding, preparation, inspection, and plotting."""
    if not args or args[0] in {"-h", "--help", "help"}:
        print(
            "usage: tsf dataset {add,list,show,search,audit,prepare,inspect,plot,"
            "convert-traffic,convert-ultratraffic,download,publish,gift-download} [args...]"
        )
        return 0
    action, rest = args[0], args[1:]
    if action in {"list", "show", "search", "audit"}:
        from moderntsf.benchmark.dataset_cards import dataset_facts, search_text
        from moderntsf.benchmark.resource_cards import audit_resource_cards, dataset_records

        records = dataset_records(ROOT)
        if action == "list":
            if rest not in ([], ["--json"]):
                print("usage: tsf dataset list [--json]", file=sys.stderr)
                return 2
            payload = [_dataset_record_payload(record) for record in records]
            if rest == ["--json"]:
                _print(payload)
            else:
                for record in payload:
                    modes = ",".join(record["task_modes"])
                    print(f"{record['name']}\t{record['loader']}\t{modes}\t{record['alias']}")
            return 0
        if action == "show":
            if len(rest) != 1:
                print("usage: tsf dataset show <preset>", file=sys.stderr)
                return 2
            selected = next((record for record in records if record.name == rest[0]), None)
            if selected is None:
                family = [record for record in records if record.loader == rest[0] and record.dataset_id]
                if family:
                    facts = dataset_facts(ROOT, [rest[0]]).get(rest[0], {})
                    _print({"name": rest[0], "kind": "dataset-family", "card": f"catalog/datasets/{rest[0]}/README.md",
                            "facts": facts, "members": [record.name for record in family]})
                    return 0
                print(f"Unknown dataset preset {rest[0]!r}", file=sys.stderr)
                return 2
            _print(_dataset_record_payload(selected, dataset_facts(ROOT, [selected.name]).get(selected.name, {})))
            return 0
        if action == "search":
            parser = argparse.ArgumentParser(prog="tsf dataset search")
            parser.add_argument("query", nargs="+")
            parser.add_argument("--limit", type=int, default=10)
            parser.add_argument("--json", action="store_true")
            parsed = parser.parse_args(rest)
            if parsed.limit < 1:
                parser.error("--limit must be positive")
            terms = set(re.findall(r"[a-z0-9]+", " ".join(parsed.query).casefold()))
            matches = []
            facts = dataset_facts(ROOT)
            for record in records:
                text = search_text(record, facts.get(record.name, {}))
                matched = sorted(term for term in terms if term in text)
                named = {term for term in terms if term in f"{record.name} {record.alias}".casefold()}
                if matched:
                    payload = _dataset_record_payload(record)
                    card = facts.get(record.name, {})
                    payload.update(
                        score=len(matched) + len(named), matched_terms=matched, summary=card.get("summary", ""),
                        domain=card.get("domain", ""), frequency=card.get("frequency", ""),
                    )
                    matches.append(payload)
            matches.sort(key=lambda item: (-int(item["score"]), str(item["name"])))
            matches = matches[: parsed.limit]
            if parsed.json:
                _print(matches)
            else:
                for match in matches:
                    print(f"{match['name']}\t{match['loader']}\t{match['domain']}\t{match['frequency']}\t{match['summary']}")
            return 0
        if rest:
            print("tsf dataset audit takes no arguments", file=sys.stderr)
            return 2
        failures = [error for error in audit_resource_cards(ROOT) if "dataset" in error]
        for failure in failures:
            print(f"ERROR: {failure}")
        failing = sum(any(f"catalog/datasets/{record.name}/README.md" in item for item in failures) for record in records)
        print(f"Dataset cards: {len(records) - failing}/{len(records)} complete and current")
        return 1 if failures else 0

    scripts = {
        "add": "new_dataset.py",
        "prepare": "pre_process.py",
        "inspect": "dataset_characteristics.py",
        "plot": "visual_data.py",
        "convert-traffic": "convert_traffic.py",
        "gift-download": "gift_eval_download.py",
    }
    if action in {"download", "publish"}:
        return _hub_dataset_command(action, rest)
    if action == "convert-ultratraffic":
        from moderntsf.data.prepare.ultratraffic import main as convert_ultratraffic

        return convert_ultratraffic(rest)
    script = scripts.get(action)
    if script is None:
        print(f"unknown dataset action: {action!r}", file=sys.stderr)
        return 2
    return passthrough(script, rest)


def _hub_dataset_command(action: str, rest: list[str]) -> int:
    """Download published preset files, or publish local ones (maintainers)."""
    from moderntsf import hub

    parser = argparse.ArgumentParser(prog=f"tsf dataset {action}")
    parser.add_argument("presets", nargs="*")
    parser.add_argument("--root", type=Path, default=Path("dataset"),
                        help="local dataset root (default: ./dataset)")
    if action == "download":
        parser.add_argument("--all", action="store_true", help="every published preset")
        parser.add_argument("--list", action="store_true", help="list published presets")
        parser.add_argument("--check", action="store_true",
                            help="check that every pinned file still resolves (no download)")
    else:
        parser.add_argument("--repo", default=hub.DEFAULT_STATIC_REPO)
        parser.add_argument("--create", action="store_true", help="create the repo if missing")
        parser.add_argument("--private", action="store_true", help="create the repo as private")
        parser.add_argument("--path", action="append", default=[], dest="paths",
                            help="also publish a whole subtree of the root (repeatable)")
    parsed = parser.parse_args(rest)
    try:
        if action == "publish":
            if not parsed.presets and not parsed.paths:
                parser.error("name presets or --path subtrees to publish")
            revisions = hub.publish_presets(parsed.presets, parsed.root, parsed.repo,
                                            paths=tuple(parsed.paths), create=parsed.create, private=parsed.private,
                                            root=ROOT)
            for preset, revision in revisions.items():
                print(f"{preset}\t{revision}")
            print("Updated configs/hub/datasets.json; commit it with the release.")
            return 0
        if parsed.check:
            from moderntsf.hub.datasets import check_manifest, load_manifest

            issues = check_manifest(ROOT)
            for issue in issues:
                print(f"ERROR: {issue}")
            total = len(load_manifest(ROOT)["files"])
            print(f"Pinned dataset files: {total - len(issues)}/{total} reachable")
            return 1 if issues else 0
        published = hub.available_presets(ROOT)
        if parsed.list:
            for preset in published:
                print(preset)
            return 0
        presets = published if parsed.all else parsed.presets
        if not presets:
            parser.error("name presets, or pass --all or --list")
        for preset in presets:
            files = hub.fetch_preset(preset, parsed.root, ROOT)
            print(f"{preset}\t{len(files)} file(s) verified under {parsed.root}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def result_command(args: list[str]) -> int:
    """Route result aggregation, ranking, plotting, reporting, and visualization."""
    if not args or args[0] in {"-h", "--help", "help"}:
        print("usage: tsf result {aggregate,rank,plot,report,predictions} [args...]")
        return 0
    action, rest = args[0], args[1:]
    scripts = {
        "aggregate": "aggregate_results.py",
        "rank": "rank_models.py",
        "plot": "plot_bubble.py",
        "report": "report.py",
        "predictions": "visualize_predictions.py",
    }
    script = scripts.get(action)
    if script is None:
        print(f"unknown result action: {action!r}", file=sys.stderr)
        return 2
    return passthrough(script, rest)

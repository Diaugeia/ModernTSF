"""Dataset and result resource command routing behind the public CLI."""

from __future__ import annotations

import argparse
import json
import re
import sys

from moderntsf.benchmark.command_runtime import ROOT, passthrough


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _dataset_record_payload(record: object) -> dict[str, object]:
    from dataclasses import asdict

    payload = asdict(record)
    payload["card"] = f"catalog/datasets/{record.name}/README.md"
    return payload


def dataset_command(args: list[str]) -> int:
    """Route dataset scaffolding, preparation, inspection, and plotting."""
    if not args or args[0] in {"-h", "--help", "help"}:
        print(
            "usage: tsf dataset {add,list,show,search,audit,prepare,inspect,plot,"
            "convert-traffic,gift-download} [args...]"
        )
        return 0
    action, rest = args[0], args[1:]
    if action in {"list", "show", "search", "audit"}:
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
                print(f"Unknown dataset preset {rest[0]!r}", file=sys.stderr)
                return 2
            _print(_dataset_record_payload(selected))
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
            for record in records:
                text = " ".join(
                    (record.name, record.alias, record.loader, record.dataset_id or record.path, record.track)
                ).casefold()
                matched = sorted(term for term in terms if term in text)
                if matched:
                    payload = _dataset_record_payload(record)
                    payload.update(score=len(matched), matched_terms=matched)
                    matches.append(payload)
            matches.sort(key=lambda item: (-int(item["score"]), str(item["name"])))
            matches = matches[: parsed.limit]
            if parsed.json:
                _print(matches)
            else:
                for match in matches:
                    print(f"{match['name']}\t{match['loader']}\t{match['alias']}")
            return 0
        if rest:
            print("tsf dataset audit takes no arguments", file=sys.stderr)
            return 2
        failures = [error for error in audit_resource_cards(ROOT) if "dataset" in error]
        for failure in failures:
            print(f"ERROR: {failure}")
        print(f"Dataset cards: {len(records) - len(failures)}/{len(records)} current")
        return 1 if failures else 0

    scripts = {
        "add": "new_dataset.py",
        "prepare": "pre_process.py",
        "inspect": "dataset_characteristics.py",
        "plot": "visual_data.py",
        "convert-traffic": "convert_traffic.py",
        "gift-download": "gift_eval_download.py",
    }
    script = scripts.get(action)
    if script is None:
        print(f"unknown dataset action: {action!r}", file=sys.stderr)
        return 2
    return passthrough(script, rest)


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

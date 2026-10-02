"""Public CLI for inspecting and rendering harness-neutral Agent tasks."""

from __future__ import annotations

import argparse
import json
import sys

from tsflab.tsf_core.agent_tasks import (
    AgentTaskError,
    audit_tasks,
    list_tasks,
    load_task,
    render_task,
    render_text,
)
from tsflab.benchmark.research_round import ResearchRoundError


def _json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _render_parser(action: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"tsf agent task {action}")
    parser.add_argument("name")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--json", action="store_true")
    return parser


def _supplied(items: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in items:
        if "=" not in item or not item.split("=", 1)[0]:
            raise AgentTaskError("--set requires KEY=VALUE")
        key, value = item.split("=", 1)
        values[key] = value
    return values


def agent_command(args: list[str]) -> int:
    """Route task inspection, rendering, and research-round preparation."""
    if not args or args[0] in {"-h", "--help", "help"}:
        print(
            "usage: tsf agent task {list,show,render,start,validate} [args...]\n"
            "start prepares a research round and prompt; it does not dispatch an Agent"
        )
        return 0
    if args[0] != "task":
        print(
            "usage: tsf agent task {list,show,render,start,validate} [args...]",
            file=sys.stderr,
        )
        return 2
    rest = args[1:]
    if not rest or rest[0] in {"-h", "--help", "help"}:
        print(
            "usage: tsf agent task {list,show,render,start,validate} [args...]\n"
            "start prepares a research round and prompt; it does not dispatch an Agent"
        )
        return 0
    action, tail = rest[0], rest[1:]
    try:
        if action == "list":
            if tail not in ([], ["--json"]):
                raise AgentTaskError("usage: tsf agent task list [--json]")
            records = list_tasks()
            if tail == ["--json"]:
                _json(records)
            else:
                for record in records:
                    print(f"{record['name']}: {record['summary']}")
            return 0
        if action == "show":
            if len(tail) != 1:
                raise AgentTaskError("usage: tsf agent task show <name>")
            _json(load_task(tail[0]))
            return 0
        if action == "validate":
            if tail:
                for name in tail:
                    load_task(name)
                print(f"Agent tasks OK: {len(tail)} selected")
                return 0
            errors = audit_tasks()
            if errors:
                for error in errors:
                    print(f"ERROR: {error}")
                return 1
            print(f"Agent tasks OK: {len(list_tasks())} templates")
            return 0
        if action in {"render", "start"}:
            parsed = _render_parser(action).parse_args(tail)
            payload = render_task(parsed.name, _supplied(parsed.set))
            if action == "start":
                from tsflab.benchmark.infra.research import prepare_task

                payload = prepare_task(parsed.name, _supplied(parsed.set), persist=True)
                round_state = payload["round"]
                prompt_path = payload["prompt_path"]
                if not parsed.json:
                    print(
                        f"Prepared research round: {round_state['id']}\n"
                        f"Prompt: {prompt_path}\n\n",
                        end="",
                    )
                    print(render_text(payload["task"]), end="")
                    return 0
            _json(payload) if parsed.json else print(render_text(payload), end="")
            return 0
    except (AgentTaskError, ResearchRoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"unknown Agent task action: {action}", file=sys.stderr)
    return 2

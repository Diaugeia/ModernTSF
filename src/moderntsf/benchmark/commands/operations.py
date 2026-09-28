"""Optional operations, discovered through tsf interface."""

import argparse

from moderntsf.benchmark.command_output import publish
import json


def operations_command(args):
    parser = argparse.ArgumentParser(prog=f"tsf {args[0]}")
    parser.add_argument("action")
    parser.add_argument("directory")
    parser.add_argument("--policy")
    parser.add_argument("--json", action="store_true")
    if args[0] == "queue":
        parser.add_argument("--run")
        parser.add_argument("--executor", help="Importable module:callable for this job")
        parser.add_argument("--priority", type=int, default=0)
        parser.add_argument("--slots", type=int, default=1)
        parser.add_argument("--once", action="store_true")
    elif args[0] == "usage":
        parser.add_argument("--operation")
        parser.add_argument("--tokens", type=int, default=0)
        parser.add_argument("--cost-usd", type=float, default=0)
    elif args[0] == "storage":
        parser.add_argument("--apply", action="store_true")
    else:
        parser.add_argument("--partition")
        parser.add_argument("--account")
        parser.add_argument("--gpus", type=int, default=0)
        parser.add_argument("--minutes", type=int, default=60)
    options = parser.parse_args(args[1:])
    try:
        from moderntsf.benchmark.infra.policy import load_policy

        policy = load_policy(options.policy)
        if args[0] == "queue":
            from moderntsf.benchmark.infra.queue import enqueue, jobs, work, cancel_job

            if options.action == "add":
                if not options.run:
                    raise ValueError("queue add requires --run <prepared directory>")
                result = enqueue(
                    options.directory, options.run, priority=options.priority, executor=options.executor
                )
            elif options.action == "cancel":
                result = cancel_job(options.directory)
            elif options.action == "status":
                result = jobs(options.directory)
            elif options.action == "work":
                result = work(options.directory, once=options.once, slots=options.slots)
            else:
                raise ValueError("queue action must be add, status, work, or cancel")
        elif args[0] == "usage":
            from moderntsf.benchmark.infra.accounting import account

            result = account(
                options.directory,
                options.action,
                options.operation,
                tokens=options.tokens,
                cost_usd=options.cost_usd,
                budget=policy.budget,
            )
        elif args[0] == "storage":
            from moderntsf.benchmark.infra.retention import cleanup, storage_status

            if options.action == "status":
                result = storage_status(options.directory, policy)
            elif options.action == "cleanup":
                result = cleanup(options.directory, policy, apply=options.apply)
            else:
                raise ValueError("storage action must be status or cleanup")
        else:
            from moderntsf.benchmark.infra.slurm import slurm

            result = slurm(
                options.directory,
                options.action,
                partition=options.partition,
                account=options.account,
                gpus=options.gpus,
                minutes=options.minutes,
            )
        publish(result)
        print(json.dumps(result, indent=2))
        return 0 if not isinstance(result, dict) or result.get("ok", True) else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
        )
        return 2

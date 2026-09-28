"""Unified executable verification entrypoint for local model implementations."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import UTC, datetime
import json
import platform
from pathlib import Path
import subprocess
import sys

from moderntsf.tsf_core.paths import repository_root, require_checkout


_RUNTIME_CHECKS = (
    "construction",
    "forward",
    "backward",
    "finite_outputs",
    "active_parameter_gradients",
    "state_dict_round_trip",
    "cpu",
    "batch_size_boundary",
    "sequence_length_boundary",
    "input_contract",
)


def _strict_contract(name: str) -> dict[str, str] | None:
    from moderntsf.benchmark.model_contracts import audit_model_contracts

    failures = audit_model_contracts([name], strict=True)
    if not failures:
        return None
    failure = failures[0]
    return {"stage": failure.stage, "error": failure.error}


def _declared_test(test: str | None) -> dict[str, object] | None:
    """Execute one manifest-declared unittest and retain a compact trace."""
    if test is None:
        return None
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", test],
        cwd=repository_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    return {
        "passed": completed.returncode == 0,
        "command": f"{sys.executable} -m pytest -q {test}",
        "test": test,
        "output": output[-2000:],
    }


def _load_existing(path: Path):
    from moderntsf.benchmark.verification import VerificationEvidence

    if not path.is_file():
        return None
    try:
        return VerificationEvidence.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _materially_changed(existing, candidate) -> bool:
    """Ignore only the wall-clock time when every verification fact is equal."""
    if existing is None:
        return True
    existing_payload = existing.model_dump(mode="json", exclude={"verified_at"})
    candidate_payload = candidate.model_dump(mode="json", exclude={"verified_at"})
    return existing_payload != candidate_payload


def _check(status: str, evidence: list[str], **metrics: object) -> dict[str, object]:
    payload: dict[str, object] = {"status": status, "evidence": evidence}
    if metrics:
        payload["metrics"] = metrics
    return payload


def _environment() -> dict[str, object]:
    import numpy
    import torch

    return {
        "python": platform.python_version(),
        "framework": f"torch {torch.__version__}",
        "dependencies": {"numpy": numpy.__version__, "torch": torch.__version__},
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "float32",
        "deterministic": {"num_threads": 1, "seed": 104729},
    }


def _execute(names: list[str], jobs: int) -> dict[str, dict[str, str] | None]:
    """Run declared checks and atomically refresh canonical evidence."""
    from moderntsf.benchmark.catalog_metadata import model_records
    from moderntsf.benchmark.verification import (
        VerificationEvidence,
        evidence_state,
        load_manifest,
        rebuild_index,
        write_evidence,
    )
    from moderntsf.benchmark.verification_common import verification_subject_sha256

    root = repository_root()
    fields = {str(record["name"]): record for record in model_records(root)}
    unknown = sorted(set(names) - fields.keys())
    if unknown:
        raise ValueError(f"unknown model(s): {', '.join(unknown)}")
    manifest = load_manifest(root, set(fields))
    if jobs == 1:
        contracts = {name: _strict_contract(name) for name in names}
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            contracts = dict(zip(names, executor.map(_strict_contract, names)))

    declared_tests = sorted(
        {
            test
            for name in names
            for test in (
                manifest.models[name].test,
                manifest.models[name].reference_test,
            )
            if test is not None
        }
    )
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        test_cache: dict[str | None, dict[str, object] | None] = {
            test: result
            for test, result in zip(declared_tests, executor.map(_declared_test, declared_tests))
        }
    test_cache[None] = None

    environment = _environment()
    verified_at = datetime.now(UTC)
    for name in names:
        record = fields[name]
        declaration = manifest.models[name]
        target = root / "verification" / "evidence" / f"{name}.json"
        existing = _load_existing(target)
        current = evidence_state(root, name, record).current
        contract = contracts[name]
        runtime_status = "passed" if contract is None else "failed"
        runtime_evidence = [f"uv run tsf verify model {name}"]
        runtime_metrics: dict[str, object] = {"device": "cpu"}
        if contract is not None:
            runtime_metrics.update(contract)

        paper_result = test_cache[declaration.test]
        if paper_result is not None:
            paper_status = "passed" if paper_result["passed"] else "failed"
            paper_check = _check(
                paper_status,
                [str(paper_result["test"])],
                profile=declaration.profile,
                structure_count=len(declaration.structure),
            )
        elif existing is not None and current:
            paper_check = existing.checks.paper_structure.model_dump(mode="json")
        else:
            paper_check = _check(
                "failed",
                ["verification/models.toml"],
                reason="stale model has no declared paper-structure test",
            )

        equation_check = dict(paper_check)
        codebase = record.get("codebase")
        reference_result = test_cache[declaration.reference_test]
        if codebase is None:
            reference_check: dict[str, object] = {
                "status": "not-applicable",
                "evidence": [],
                "metrics": {"reason": "no official codebase"},
            }
        elif reference_result is not None:
            reference_status = "passed" if reference_result["passed"] else "failed"
            inspected_sources = declaration.reference_sources or [
                f"{codebase['url']}@{codebase['revision']}"
            ]
            reference_check = _check(
                reference_status,
                [
                    *inspected_sources,
                    str(reference_result["test"]),
                ],
                profile=declaration.profile,
                inspected_files=len(inspected_sources),
            )
        elif existing is not None and current and existing.codebase is not None:
            reference_check = existing.checks.reference_comparison.model_dump(mode="json")
        else:
            reference_check = _check(
                "failed",
                ["verification/models.toml"],
                reason="stale model has no declared reference-comparison test",
            )

        checks: dict[str, object] = {
            "paper_structure": paper_check,
            "equations": equation_check,
            **{
                check_name: _check(runtime_status, runtime_evidence, **runtime_metrics)
                for check_name in _RUNTIME_CHECKS
            },
            "reference_comparison": reference_check,
        }
        if "inference-only" in set(record.get("capabilities", ())):
            for check_name in ("backward", "active_parameter_gradients"):
                checks[check_name] = {
                    "status": "not-applicable",
                    "evidence": [],
                    "metrics": {
                        "reason": "official runtime is declared inference-only"
                    },
                }
        status = "passed" if all(
            check["status"] in {"passed", "not-applicable"} for check in checks.values()
        ) else "failed"
        commands = [f"uv run tsf verify model {name}"]
        for result in (paper_result, reference_result):
            if result is not None and str(result["command"]) not in commands:
                commands.append(str(result["command"]))
        payload = {
            "schema_version": 1,
            "model": name,
            "status": status,
            "verified_at": verified_at,
            "subject_sha256": verification_subject_sha256(root, record),
            "paper": record["paper"],
            "codebase": codebase,
            "checks": checks,
            "environment": environment,
            "commands": commands,
        }
        candidate = VerificationEvidence.model_validate(payload)
        if not current or _materially_changed(existing, candidate):
            write_evidence(root, candidate)
    rebuild_index(root)
    return contracts


def _records(
    names: list[str],
    jobs: int,
    run_contracts: bool,
    known_contracts: dict[str, dict[str, str] | None] | None = None,
) -> list[dict[str, object]]:
    from moderntsf.benchmark.catalog_metadata import model_records
    from moderntsf.benchmark.verification import evidence_state, load_manifest

    root = repository_root()
    fields = {str(record["name"]): record for record in model_records(root)}
    unknown = sorted(set(names) - fields.keys())
    if unknown:
        raise ValueError(f"unknown model(s): {', '.join(unknown)}")
    manifest = load_manifest(root, set(fields))
    contract_failures: dict[str, dict[str, str] | None] = known_contracts or {}
    if run_contracts:
        if jobs == 1:
            contract_failures = {name: _strict_contract(name) for name in names}
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                contract_failures = dict(zip(names, executor.map(_strict_contract, names)))
    records = []
    for name in names:
        state = evidence_state(root, name, fields[name])
        contract = contract_failures.get(name)
        status = "failed" if contract is not None else state.status
        records.append(
            {
                "model": name,
                "profile": manifest.models[name].profile,
                "status": status,
                "evidence_status": state.status,
                "current": state.current,
                "evidence": state.evidence,
                "detail": state.detail,
                "contract": "failed" if contract is not None else "passed" if (run_contracts or known_contracts is not None) else "not-run",
                "contract_failure": contract,
            }
        )
    return records


def _emit(records: list[dict[str, object]], as_json: bool) -> int:
    if as_json:
        print(json.dumps(records, indent=2, sort_keys=True))
    else:
        for record in records:
            detail = f": {record['detail']}" if record["detail"] else ""
            print(f"{str(record['status']).upper()} {record['model']}{detail}")
        passed = sum(record["status"] == "passed" for record in records)
        print(f"Verification: {passed}/{len(records)} models passed")
    return 0 if all(record["status"] == "passed" for record in records) else 1


def verification_command(args: list[str]) -> int:
    """Run or inspect the final route-neutral model verification contract."""
    if not args or args[0] in {"-h", "--help", "help"}:
        print(
            "usage: tsf verify {model,stale,all,index} [args...]\n"
            "       tsf verify model <Name...> [--jobs N] [--json]\n"
            "       tsf verify stale [--json]\n"
            "       tsf verify all [--jobs N] [--json]\n"
            "       tsf verify index"
        )
        return 0
    action, rest = args[0], args[1:]
    if action == "index":
        if rest:
            print("tsf verify index takes no arguments", file=sys.stderr)
            return 2
        from moderntsf.benchmark.verification import rebuild_index

        try:
            root = require_checkout("tsf verify index")
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        index = rebuild_index(root)
        print(f"Indexed verification evidence for {len(index.models)} models")
        return 0
    if action not in {"model", "stale", "all"}:
        print("usage: tsf verify {model,stale,all,index} [args...]", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(prog=f"tsf verify {action}")
    if action == "model":
        parser.add_argument("names", nargs="+")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    parsed = parser.parse_args(rest)
    if parsed.jobs < 1:
        parser.error("--jobs must be positive")

    from moderntsf.benchmark.catalog_metadata import model_records

    catalog_names = [str(record["name"]) for record in model_records(repository_root())]
    names = parsed.names if action == "model" else catalog_names
    try:
        contracts = None
        if action != "stale":
            require_checkout(f"tsf verify {action}")
            contracts = _execute(names, parsed.jobs)
        records = _records(names, parsed.jobs, False, contracts)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if action == "stale":
        records = [record for record in records if not record["current"]]
        if parsed.json:
            print(json.dumps(records, indent=2, sort_keys=True))
        else:
            for record in records:
                print(f"{str(record['evidence_status']).upper()} {record['model']}: {record['detail']}")
            print(f"Stale verification evidence: {len(records)} models")
        return 1 if records else 0
    return _emit(records, parsed.json)

"""Executable module boundaries and independent composition contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import threading

from tsflab.experiments.infra.api import (
    Budget,
    Storage,
    Tracker,
    UsageLedger,
    FileCancellation,
    any_cancelled,
    describe_modules,
    storage_status,
)


def test_leaf_services_work_without_training_or_orchestration_imports(tmp_path):
    script = """
import importlib.abc
import sys
from pathlib import Path
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = ('torch', 'wandb', 'tensorboard', 'tsflab.experiments.runner',
                     'tsflab.experiments.infra.execution', 'tsflab.experiments.infra.environment',
                     'tsflab.experiments.infra.runs')
        if any(fullname == name or fullname.startswith(name + '.') for name in forbidden):
            raise AssertionError('unwanted dependency: ' + fullname)
sys.meta_path.insert(0, Boundary())
from tsflab.experiments.infra.api import (Tracker, UsageLedger, Budget, Storage,
    Resources, lease_gpus, storage_status, describe_modules, enqueue, run_job)
root = Path(sys.argv[1])
with Tracker(root / 'metrics', 'standalone') as tracker:
    tracker.log({'loss': .25}, 1)
ledger = UsageLedger(root / 'ledger', Budget(max_tokens=10))
ledger.reserve('call', tokens=10)
ledger.settle('call', tokens=4)
assert ledger.status()['totals']['tokens'] == 4
assert storage_status(root / 'metrics', Storage(max_run_gb=1))['ok']
with lease_gpus(Resources(gpus=['0']), directory=root / 'leases',
    inventory=lambda: [{'index':'0', 'uuid':'test', 'free_mb':100}]) as assigned:
    assert assigned == ['test']
item = enqueue(root / 'queue', root / 'input', validate=lambda path: None)
run_job(root / 'queue' / item['id'], executor=lambda path, cancelled: {'ok': not cancelled()})
assert describe_modules()['modules']
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_tracking_has_explicit_idempotent_lifecycle(tmp_path):
    path = tmp_path / "new" / "events"
    with Tracker(path, "test") as tracker:
        tracker.log({"loss": 0.5}, 2)
    tracker.close()
    assert json.loads((path / "events.jsonl").read_text())["metrics"]["loss"] == 0.5
    import pytest

    with pytest.raises(RuntimeError, match="closed"):
        tracker.log({"loss": 0}, 3)


def test_injected_queue_execution_has_independent_cancellation(tmp_path):
    from tsflab.experiments.infra.queue import enqueue, run_job, cancel_job, jobs

    root = tmp_path / "queue"
    items = [
        enqueue(root, tmp_path / name, validate=lambda path: None)
        for name in ("a", "b")
    ]
    before = dict(os.environ)
    entered = threading.Barrier(3)
    released = threading.Event()
    seen = {}

    def executor(directory, *, cancelled):
        entered.wait(timeout=10)
        assert released.wait(timeout=10)
        seen[Path(directory).name] = cancelled()
        return {"ok": not cancelled()}

    with ThreadPoolExecutor(2) as pool:
        futures = [
            pool.submit(run_job, root / item["id"], executor=executor) for item in items
        ]
        entered.wait(timeout=10)
        cancel_job(root / items[0]["id"])
        released.set()
        for future in futures:
            future.result(timeout=10)
    assert seen == {"a": True, "b": False}
    assert {Path(j["directory"]).name: j["status"] for j in jobs(root)} == {
        "a": "cancelled",
        "b": "succeeded",
    }
    assert dict(os.environ) == before


def test_module_discovery_matches_real_exports_and_schema():
    import importlib

    for module in describe_modules()["modules"]:
        loaded = importlib.import_module(module["import"])
        for name in module["exports"]:
            assert callable(getattr(loaded, name)), (module["name"], name)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tsflab.cli.main",
            "agent",
            "interface",
            "schema",
            "--module",
            "storage",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert set(json.loads(result.stdout)["properties"]) == {
        "max_run_gb",
        "keep_epoch_checkpoints",
    }


def test_cancellation_composition_is_read_only_until_requested(tmp_path):
    token = FileCancellation(tmp_path / "nested" / "cancel")
    combined = any_cancelled(lambda: False, token)
    assert not combined()
    assert not token.path.parent.exists()
    token.request()
    assert combined()


def test_storage_scope_accepts_local_options_and_rejects_missing_path(tmp_path):
    import pytest

    assert storage_status(tmp_path, Storage())["ok"]
    with pytest.raises(ValueError, match="existing directory"):
        storage_status(tmp_path / "missing")


def test_ledger_facade_keeps_limits_across_instances(tmp_path):
    import pytest

    first = UsageLedger(tmp_path, Budget(max_tokens=10))
    first.reserve("one", tokens=8)
    with pytest.raises(ValueError, match="exhausted"):
        UsageLedger(tmp_path).reserve("two", tokens=3)
    first.settle("one", tokens=1)
    UsageLedger(tmp_path).reserve("two", tokens=3)


def test_agent_task_preparation_is_optional_and_nonpersistent(tmp_path, monkeypatch):
    from tsflab.experiments.infra.api import prepare_task

    target = tmp_path / "workspace"
    monkeypatch.setenv("TSF_WORK_DIR", str(target))
    prepared = prepare_task("autoresearch", {"question": "test", "max_runs": "1"})
    assert prepared["round"] is None and prepared["prompt_path"] is None
    assert prepared["dispatch"] == "not-performed"
    assert prepared["task"]["budget"]["max_runs"] == 1
    assert not target.exists()


def test_agent_api_and_cli_share_persistent_budget_service(
    tmp_path, monkeypatch, capsys
):
    import pytest
    from tsflab.experiments.infra.api import prepare_task, load_round
    from tsflab.research.rounds import claim_run, ResearchRoundError
    from tsflab.cli.main import main

    monkeypatch.setenv("TSF_WORK_DIR", str(tmp_path))
    prepared = prepare_task(
        "autoresearch", {"question": "test", "max_runs": "1"}, persist=True
    )
    round_id = prepared["round"]["id"]
    assert Path(prepared["prompt_path"]).exists()
    claim_run(round_id, {"run_id": "first"})
    with pytest.raises(ResearchRoundError):
        claim_run(round_id, {"run_id": "second"})
    assert (
        main(
            [
                "agent",
                "task",
                "start",
                "autoresearch",
                "--set",
                "question=test",
                "--set",
                "max_runs=1",
                "--json",
            ]
        )
        == 0
    )
    cli = json.loads(capsys.readouterr().out)
    assert cli["task"] == prepared["task"]
    assert load_round(cli["round"]["id"])["budget"] == prepared["round"]["budget"]
    assert cli["dispatch"] == "not-performed"

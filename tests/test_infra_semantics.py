"""Regression contracts for budgets, Agent-owned iterations and adapter parity."""

import json
import subprocess
import sys
import time

import pytest

from tsflab.benchmark.infra.policy import ExecutionPolicy


def test_budget_has_one_validated_source_before_persistence(tmp_path, monkeypatch):
    from tsflab.benchmark.research_round import (
        create_round,
        claim_run,
        load_round,
        ResearchRoundError,
    )

    monkeypatch.setenv("TSF_WORK_DIR", str(tmp_path))
    for kwargs in (
        {"max_runs": 2, "budget": {"max_runs": 3}},
        {"budget": {"max_runs": -1}},
        {"budget": {"max_runs": True}},
        {"budget": {"max_gpu_hours": float("nan")}},
    ):
        with pytest.raises(ResearchRoundError):
            create_round(task="test", goal="test", **kwargs)
    assert not list(tmp_path.iterdir())
    state = create_round(task="test", goal="test", budget={"max_runs": 1})
    assert state["max_runs"] == state["budget"]["max_runs"] == 1
    claim_run(state["id"], {"run_id": "one"})
    with pytest.raises(ResearchRoundError, match="exhausted"):
        claim_run(state["id"], {"run_id": "two"})
    assert load_round(state["id"])["runs_used"] == 1


def test_matrix_preparation_does_not_define_research_iterations(tmp_path, monkeypatch):
    from tsflab.benchmark.config.loader import load_config
    from tsflab.benchmark.infra.execution import prepare_sweep
    from tsflab.benchmark.research_round import (
        create_round,
        load_round,
        claim_iteration,
        ResearchRoundError,
    )

    monkeypatch.setenv("TSF_WORK_DIR", str(tmp_path))
    state = create_round(task="test", goal="test", budget={"max_iterations": 1})
    loaded = load_config("configs/runs/smoke_crib.toml")
    loaded[0].config.experiment.work_dir = str(tmp_path)
    prepare_sweep(loaded, ExecutionPolicy(), state["id"])
    assert load_round(state["id"])["iterations_used"] == 0
    claim_iteration(state["id"], operation="hypothesis-a")
    prepare_sweep(loaded, ExecutionPolicy(), state["id"])
    claim_iteration(state["id"], operation="hypothesis-a")
    assert load_round(state["id"])["iterations_used"] == 1
    with pytest.raises(ResearchRoundError, match="exhausted"):
        claim_iteration(state["id"], operation="hypothesis-b")


def test_preflight_returns_resolved_copy_and_preserves_input(tmp_path, monkeypatch):
    from tsflab.benchmark.config.loader import load_config
    from tsflab.benchmark.infra import execution

    config = load_config("configs/runs/smoke_crib.toml")[0].config
    config.experiment.runtime.device = "cuda"
    config.experiment.runtime.device_ids = [0]
    policy = ExecutionPolicy()
    original = policy.model_dump()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        "tsflab.benchmark.infra.environment.gpu_inventory", lambda: [{"uuid": "gpu-test"}]
    )
    monkeypatch.setattr(execution, "audit_environment", lambda *a, **k: {"checks": []})
    report = execution.preflight([config], policy)
    assert report["ok"]
    assert report["resolved_policy"]["resources"]["gpus"] == ["gpu-test"]
    assert policy.model_dump() == original
    report["resolved_policy"]["resources"]["gpus"].clear()
    assert policy.model_dump() == original


def test_model_pretraining_works_without_infrastructure_import():
    code = """
import importlib.abc
import sys
import torch
from tsflab.models.latenttsf.model import Model
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('tsflab.benchmark.infra'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
model = Model(4, 2, 1, d_model=2, d_ff=3, kernel_size=3, ae_train_epochs=1)
model.pretrain([(torch.ones(2,4,1),)], torch.device('cpu'))
assert model._autoencoder_frozen
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_same_executor_contract_in_process_and_detached(tmp_path, monkeypatch):
    from tsflab.benchmark.infra.queue import enqueue, run_job, work, jobs

    module = tmp_path / "local_test_executor.py"
    module.write_text(
        'def execute(directory, *, cancelled):\n    return {"ok": not cancelled(), "value": 17}\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    import os

    monkeypatch.setenv(
        "PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    root = tmp_path / "queue"
    first = enqueue(
        root,
        tmp_path / "one",
        validate=lambda p: None,
        executor="local_test_executor:execute",
    )
    local = run_job(root / first["id"])
    second = enqueue(
        root,
        tmp_path / "two",
        validate=lambda p: None,
        executor="local_test_executor:execute",
    )
    work(root, once=True)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        remote = next(job for job in jobs(root) if job["id"] == second["id"])
        if remote["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert remote["status"] == local["status"] == "succeeded", remote
    assert remote["result"] == local["result"]


def test_executor_contract_errors_are_structured(tmp_path):
    from tsflab.benchmark.infra.queue import enqueue, run_job

    item = enqueue(tmp_path, tmp_path / "input", validate=lambda p: None)
    state = run_job(tmp_path / item["id"], executor=lambda *a, **k: {"ok": "yes"})
    assert state["status"] == "failed"
    assert state["error"]["type"] == "ContractError"


def test_cli_envelope_calls_route_directly_and_shares_result_shape(monkeypatch, capsys):
    from tsflab.benchmark.cli import main
    from tsflab.benchmark.infra.results import invoke

    def no_subprocess(*a, **k):
        raise AssertionError("unexpected CLI subprocess")

    monkeypatch.setattr(subprocess, "run", no_subprocess)
    assert main(["--format", "json", "agent", "interface", "schema", "--module", "storage"]) == 0
    actual = json.loads(capsys.readouterr().out)
    expected = invoke(lambda: actual["data"]).to_dict()
    assert {key: actual[key] for key in expected} == expected
    assert main(["--format", "json", "run", "--bad-flag"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]

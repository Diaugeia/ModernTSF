"""Behavioral regression tests for execution, recovery, and scientific reporting."""

import json
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from moderntsf.benchmark.infra.policy import ExecutionPolicy
from moderntsf.benchmark.infra.comparison import compare_rows
from moderntsf.benchmark.runner.trainer import train


class Forecaster(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x, marks, decoder, decoder_marks):
        return self.linear(self.dropout(x.transpose(1, 2))).transpose(1, 2)


class InterruptAfterEpoch:
    def start(self, step):
        pass

    def log(self, metrics, step):
        if step == 1:
            raise RuntimeError("simulated interruption after checkpoint commit")


def training_case(directory, *, resume=False, tracker=None):
    torch.set_num_threads(1)
    torch.manual_seed(431)
    model = Forecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(
        torch.arange(64).reshape(16, 4, 1).float() / 64,
        torch.arange(32).reshape(16, 2, 1).float() / 32,
        torch.zeros(16, 4, 1),
        torch.zeros(16, 2, 1),
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    result = train(
        model,
        loader,
        loader,
        torch.device("cpu"),
        epochs=3,
        patience=10,
        loss_name="mse",
        loss_params={},
        optimizer=optimizer,
        lradj="exponential",
        base_lr=0.01,
        total_epochs=3,
        label_len=0,
        pred_len=2,
        features="M",
        use_amp=False,
        checkpoint_dir=str(directory),
        checkpoint_cfg=SimpleNamespace(strategy="best", save_k=1),
        resume=resume,
        tracker=tracker,
    )
    return model, optimizer, result


def test_resume_matches_uninterrupted_training_including_rng_and_optimizer(tmp_path):
    expected_model, expected_optimizer, _ = training_case(tmp_path / "full")
    with pytest.raises(RuntimeError, match="simulated"):
        training_case(tmp_path / "resumed", tracker=InterruptAfterEpoch())
    model, optimizer, _ = training_case(tmp_path / "resumed", resume=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            value, expected_model.state_dict()[key], rtol=0, atol=0
        )
    for key, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value,
                expected_optimizer.state_dict()["state"][key][name],
                rtol=0,
                atol=0,
            )
    before = (tmp_path / "resumed" / "latest.pth").read_bytes()
    training_case(tmp_path / "resumed", resume=True)
    assert (tmp_path / "resumed" / "latest.pth").read_bytes() == before


def test_policy_is_optional_and_rejects_unknown_or_invalid_limits():
    policy = ExecutionPolicy()
    assert policy.tracking.wandb == "disabled" and not policy.tracking.tensorboard
    with pytest.raises(ValueError):
        ExecutionPolicy.model_validate({"budget": {"max_parallel_jobs": 0}})
    with pytest.raises(ValueError):
        ExecutionPolicy.model_validate({"budget": {"max_runz": 2}})


def test_gpu_lease_is_exclusive_and_released(tmp_path, monkeypatch):
    import moderntsf.benchmark.infra.resources as resource_module

    monkeypatch.setenv("MODERNTSF_RESOURCE_DIR", str(tmp_path))
    monkeypatch.setattr(
        resource_module,
        "gpu_inventory",
        lambda: [{"index": "3", "uuid": "GPU-test", "free_mb": "8000"}],
    )
    resources = ExecutionPolicy.model_validate(
        {"resources": {"gpus": ["3"], "wait_timeout_minutes": 0.001}}
    ).resources
    with resource_module.lease_gpus(resources) as devices:
        assert devices == ["GPU-test"]
        with pytest.raises(TimeoutError):
            with resource_module.lease_gpus(resources):
                pass
    with resource_module.lease_gpus(resources) as devices:
        assert devices == ["GPU-test"]


def test_parallel_round_budget_and_resume_do_not_double_count(tmp_path, monkeypatch):
    from moderntsf.benchmark.research_round import (
        create_round,
        claim_run,
        finish_run,
        load_round,
        ResearchRoundBusy,
    )

    monkeypatch.setenv("TSF_WORK_DIR", str(tmp_path))
    state = create_round(
        task="test", goal="bounded runs", max_runs=1, budget={"max_parallel_jobs": 1}
    )
    number = claim_run(state["id"], {"run_id": "a", "gpus": 1}, active=True)
    with pytest.raises(ResearchRoundBusy):
        claim_run(state["id"], {"run_id": "b"}, active=True)
    finish_run(state["id"], number, status="failed")
    assert claim_run(state["id"], {"run_id": "a"}, active=True) == number
    assert load_round(state["id"])["runs_used"] == 1
    finish_run(state["id"], number, status="passed")
    assert load_round(state["id"])["gpu_hours_used"] > 0


def test_csv_writes_are_concurrent_and_idempotent(tmp_path):
    from moderntsf.benchmark.utils.results import write_csv_summary
    import csv

    path = tmp_path / "performance.csv"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda i: write_csv_summary(
                    str(path), {"run_id": str(i % 4), "mse": i}
                ),
                range(24),
            )
        )
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 4 and {r["run_id"] for r in rows} == {"0", "1", "2", "3"}


def test_comparison_separates_protocols_and_exposes_missing_or_duplicate_seeds():
    def row(model, seed, protocol="p", variant="v"):
        return {
            "model": model,
            "seed": seed,
            "protocol_sha256": protocol,
            "model_variant": variant,
            "mse": 1,
            "mae": 1,
        }

    result = compare_rows(
        [
            row("a", 0),
            row("a", 1),
            row("b", 0),
            row("c", 0, "different"),
            {"model": "legacy"},
        ]
    )
    assert len(result["cohorts"]) == 2 and result["unverified_runs"] == 1
    cohort = next(c for c in result["cohorts"] if c["protocol"] == "p")
    assert cohort["leaderboard"][0]["rankable"]
    assert cohort["leaderboard"][1]["missing_seeds"] == ["1"]
    duplicated = compare_rows([row("a", 0), row("a", 0)])
    assert not duplicated["cohorts"][0]["leaderboard"][0]["rankable"]


def test_report_propagates_aggregation_failure_without_reading_stale_csv(
    tmp_path, monkeypatch
):
    from moderntsf.benchmark.commands import report

    monkeypatch.setattr(
        "sys.argv", ["report", "--dataset", "fixture", "--work-dir", str(tmp_path)]
    )
    monkeypatch.setattr(
        report,
        "_run_tool",
        lambda *args: SimpleNamespace(
            returncode=7, stderr="aggregation failed", stdout=""
        ),
    )
    assert report.main() == 7
    assert not (tmp_path / "fixture" / "report.md").exists()


def test_run_records_failure_before_data_loading_and_rejects_changed_data(
    tmp_path, monkeypatch
):
    import importlib
    from moderntsf.benchmark.config.loader import load_config
    from moderntsf.benchmark.infra.runs import prepare_run, read_run, verify_resume

    module = importlib.import_module("moderntsf.benchmark.runner.run_one")
    loaded = load_config("configs/runs/smoke_crib.toml")[0]
    config = loaded.config
    config.experiment.work_dir = str(tmp_path)
    directory = prepare_run(config, loaded.raw)
    monkeypatch.setenv("MODERNTSF_RUN_DIR", str(directory))
    monkeypatch.setattr(
        module,
        "_build_loaders",
        lambda *args: (_ for _ in ()).throw(RuntimeError("loader exploded")),
    )
    with pytest.raises(RuntimeError, match="loader exploded"):
        module.run_one(config, loaded.raw)
    record = read_run(directory)
    assert record["status"] == "failed" and record["stage"] == "data"
    assert record["attempts"][0]["status"] == "failed"
    config.task.pred_len += 1
    with pytest.raises(ValueError, match="configuration changed"):
        verify_resume(directory, config)


def test_default_tracking_has_no_optional_imports(tmp_path, monkeypatch):
    from moderntsf.benchmark.infra.tracking import Tracker
    import sys

    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setitem(sys.modules, "tensorboard", None)
    tracker = Tracker(tmp_path, "run", {}, ExecutionPolicy().tracking)
    tracker.log({"train/loss": 0.5}, 1)
    tracker.close()
    assert json.loads((tmp_path / "events.jsonl").read_text())["metrics"] == {
        "train/loss": 0.5
    }


def test_real_subprocess_sweep_and_resume_skip_completed(tmp_path):
    from moderntsf.benchmark.config.loader import load_config
    from moderntsf.benchmark.infra.execution import prepare_sweep, execute, status
    from moderntsf.benchmark.infra.runs import read_run

    loaded = load_config("configs/runs/smoke_crib.toml")
    loaded[0].config.experiment.work_dir = str(tmp_path)
    policy = ExecutionPolicy()
    directory = prepare_sweep(loaded, policy)
    result = execute(directory, policy)
    assert result["ok"], status(directory)
    path = Path(result["runs"][0]["directory"])
    assert (path / "checkpoints" / "latest.pth").is_file()
    assert (path / "attempt-1.log").is_file()
    saved = read_run(path)
    record = json.loads(
        next((tmp_path / "weather" / "CRIB" / "records").glob("*.json")).read_text()
    )
    assert record["config"]["snapshot"] and record["config"]["config_sha256"]
    resumed = execute(directory)
    assert resumed["ok"] and resumed["skipped"] == 1
    assert len(read_run(path)["attempts"]) == len(saved["attempts"]) == 1


def test_real_process_timeout_preserves_failure_and_can_resume(tmp_path):
    from moderntsf.benchmark.config.loader import load_config
    from moderntsf.benchmark.infra.execution import prepare_sweep, execute
    from moderntsf.benchmark.infra.runs import read_run

    loaded = load_config("configs/runs/smoke_crib.toml")
    loaded[0].config.experiment.work_dir = str(tmp_path)
    policy = ExecutionPolicy.model_validate({"budget": {"run_timeout_minutes": 0.001}})
    directory = prepare_sweep(loaded, policy)
    result = execute(directory, policy)
    assert not result["ok"] and result["runs"][0]["status"] == "timed_out"
    result = execute(directory, ExecutionPolicy())
    assert result["ok"]
    assert read_run(result["runs"][0]["directory"])["status"] == "succeeded"


def test_real_optional_tracking_writes_tensorboard_and_wandb_offline(tmp_path):
    pytest.importorskip("tensorboard")
    pytest.importorskip("wandb")
    from moderntsf.benchmark.infra.tracking import Tracker
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tracker = Tracker(
        tmp_path,
        "offline-fixture",
        {"seed": 17},
        ExecutionPolicy.model_validate(
            {"tracking": {"tensorboard": True, "wandb": "offline"}}
        ).tracking,
    )
    tracker.log({"train/loss": 0.25, "validation/loss": 0.5}, 1)
    tracker.close()
    accumulator = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert accumulator.Scalars("train/loss")[0].value == 0.25
    assert list((tmp_path / "wandb").rglob("*.wandb"))


def test_tracker_failure_preserves_local_events_and_closes_backend(tmp_path):
    from moderntsf.benchmark.infra.tracking import Tracker

    class BrokenWriter:
        closed = False

        def add_scalar(self, *args):
            raise RuntimeError("backend disconnected")

        def close(self):
            self.closed = True

    backend = BrokenWriter()
    tracker = Tracker(tmp_path, "failure", {}, ExecutionPolicy().tracking)
    tracker.started = True
    tracker.writer = backend
    with pytest.warns(UserWarning, match="local events"):
        tracker.log({"train/loss": 0.5}, 1)
    tracker.log({"train/loss": 0.4}, 2)
    tracker.close()
    assert backend.closed
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2


def test_planned_cells_with_no_results_remain_in_comparison():
    plan = [
        {
            "model": model,
            "model_variant": model,
            "protocol_sha256": "same",
            "seed": seed,
        }
        for model in ("a", "b")
        for seed in (0, 1)
    ]
    result = compare_rows([{**plan[0], "mse": 0.5, "mae": 0.25}], planned=plan)
    rows = result["cohorts"][0]["leaderboard"]
    assert len(rows) == 2
    assert not any(row["rankable"] for row in rows)
    assert next(row for row in rows if row["model"] == "b")["missing_seeds"] == [
        "0",
        "1",
    ]


def test_real_cancellation_and_single_run_resume_preserve_sweep_context(tmp_path):
    import time
    from moderntsf.benchmark.config.loader import load_config
    from moderntsf.benchmark.infra.execution import prepare_sweep, execute, cancel
    from moderntsf.benchmark.infra.runs import read_run

    loaded = load_config("configs/runs/smoke_crib.toml")
    loaded[0].config.experiment.work_dir = str(tmp_path)
    directory = prepare_sweep(loaded, ExecutionPolicy())
    run_path = Path(json.loads((directory / "sweep.json").read_text())["runs"][0])
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(execute, directory)
        deadline = time.monotonic() + 15
        while not (run_path / "runtime.json").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (run_path / "runtime.json").exists()
        cancel(run_path)
        result = future.result(timeout=20)
    assert result["runs"][0]["status"] == "cancelled"
    resumed = execute(run_path)
    assert resumed["directory"] == str(directory)
    assert resumed["ok"]
    assert len(read_run(run_path)["attempts"]) == 2


def test_batch_recovery_matches_uninterrupted(tmp_path, monkeypatch):
    import moderntsf.benchmark.infra.checkpoint as checkpoints

    original_train = train
    monkeypatch.setitem(
        training_case.__globals__,
        "train",
        lambda *args, **kwargs: original_train(
            *args, checkpoint_every_batches=1, **kwargs
        ),
    )
    expected, _, _ = training_case(tmp_path / "full")
    save = checkpoints.save_checkpoint

    def interrupt(*args, **kwargs):
        save(*args, **kwargs)
        if kwargs.get("progress", {}).get("next_batch") == 2:
            raise RuntimeError("batch interruption")

    monkeypatch.setattr(checkpoints, "save_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="batch interruption"):
        training_case(tmp_path / "partial")
    monkeypatch.setattr(checkpoints, "save_checkpoint", save)
    actual, _, _ = training_case(tmp_path / "partial", resume=True)
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)


def test_external_usage_reservation_is_atomic_and_idempotent(tmp_path):
    from moderntsf.benchmark.infra.accounting import account

    budget = ExecutionPolicy.model_validate(
        {"budget": {"max_tokens": 100, "max_cost_usd": 1}}
    ).budget

    def reserve(i):
        try:
            account(tmp_path, "reserve", str(i), tokens=60, cost_usd=0.6, budget=budget)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(reserve, [0, 1]))
    assert sum(results) == 1
    key = str(results.index(True))
    account(tmp_path, "settle", key, tokens=40, cost_usd=0.4)
    account(tmp_path, "settle", key, tokens=40, cost_usd=0.4)
    assert account(tmp_path, "status")["totals"]["tokens"] == 40
    with pytest.raises(ValueError, match="immutable"):
        account(tmp_path, "settle", key, tokens=20)


def test_storage_preview_protects_live_and_best_files(tmp_path):
    from moderntsf.benchmark.infra.retention import cleanup
    from moderntsf.benchmark.infra.storage import file_lock

    (tmp_path / "manifest.json").write_text("{}")
    folder = tmp_path / "checkpoints"
    folder.mkdir()
    for name in ["epoch_1.pth", "epoch_2.pth", "best_checkpoint.pth"]:
        (folder / name).write_text("weights")
    policy = ExecutionPolicy.model_validate({"storage": {"keep_epoch_checkpoints": 1}})
    plan = cleanup(tmp_path, policy)
    assert plan["files"] == [str(folder / "epoch_1.pth")]
    assert (folder / "epoch_1.pth").exists()
    with file_lock(tmp_path / ".run.lock"):
        with pytest.raises(BlockingIOError):
            cleanup(tmp_path, policy, apply=True)
    cleanup(tmp_path, policy, apply=True)
    assert (folder / "best_checkpoint.pth").exists()
    assert not (folder / "epoch_1.pth").exists()


def test_shared_gpu_limits_and_exclusive_compatibility(tmp_path, monkeypatch):
    import moderntsf.benchmark.infra.resources as resources

    monkeypatch.setenv("MODERNTSF_RESOURCE_DIR", str(tmp_path))
    monkeypatch.setattr(
        resources,
        "gpu_inventory",
        lambda: [{"index": "0", "uuid": "test", "free_mb": "8000", "total_mb": "8000"}],
    )
    policy = ExecutionPolicy.model_validate(
        {
            "resources": {
                "gpus": ["0"],
                "sharing": True,
                "memory_per_run_mb": 3000,
                "wait_timeout_minutes": 0.001,
            }
        }
    )
    with resources.lease_gpus(policy.resources):
        with resources.lease_gpus(policy.resources):
            with pytest.raises(TimeoutError):
                with resources.lease_gpus(policy.resources):
                    pass
        with pytest.raises(TimeoutError):
            with resources.lease_gpus(
                policy.resources.model_copy(update={"sharing": False})
            ):
                pass
    with resources.lease_gpus(policy.resources.model_copy(update={"sharing": False})):
        pass


def test_queue_priority_and_duplicate_submission(tmp_path, monkeypatch):
    from moderntsf.benchmark.infra.queue import enqueue, jobs

    monkeypatch.setattr("moderntsf.benchmark.infra.execution.status", lambda p: {})
    low = enqueue(tmp_path, tmp_path / "low", priority=1)
    high = enqueue(tmp_path, tmp_path / "high", priority=10)
    assert enqueue(tmp_path, tmp_path / "low")["id"] == low["id"]
    assert [j["id"] for j in jobs(tmp_path)] == [high["id"], low["id"]]


def test_slurm_uses_argument_arrays_and_retains_job_identity(tmp_path, monkeypatch):
    from moderntsf.benchmark.infra.slurm import slurm

    (tmp_path / "sweep.json").write_text("{}")
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(
            stdout="123;cluster\n" if args[0] == "sbatch" else "123|COMPLETED|0:0\n"
        )

    monkeypatch.setattr("moderntsf.benchmark.infra.slurm.subprocess.run", command)
    assert slurm(tmp_path, "submit", partition="a;echo bad")["job_id"] == "123"
    assert "a;echo bad" in calls[0]
    with pytest.raises(ValueError, match="already"):
        slurm(tmp_path, "submit")
    assert slurm(tmp_path, "status")["records"][0]["state"] == "COMPLETED"
    slurm(tmp_path, "cancel")
    assert calls[-1] == ["scancel", "--clusters", "cluster", "123"]


def test_cli_envelope_includes_parser_failures():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "moderntsf.benchmark.cli", "--format", "json", "run", "--invalid"],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert not payload["ok"] and payload["exit_code"] == 2
    assert payload["error"]["message"]


def test_pretraining_resume_matches_full_training(tmp_path, monkeypatch):
    import moderntsf.benchmark.infra.stages as stages
    from moderntsf.models.latenttsf.model import Model

    def run(path):
        torch.manual_seed(77)
        model = Model(4, 2, 1, d_model=2, d_ff=3, kernel_size=3, ae_train_epochs=2)
        loader = DataLoader(
            TensorDataset(torch.arange(32).reshape(8, 4, 1).float() / 32),
            batch_size=2,
            shuffle=True,
        )
        from functools import partial

        model.pretrain(
            loader,
            torch.device("cpu"),
            stage_runner=partial(
                stages.reconstruction_stage, checkpoint=path, every_batches=1
            ),
        )
        return model

    expected = run(tmp_path / "full.pth")
    save = stages.atomic_state

    def interrupted(path, state):
        save(path, state)
        if state["next_batch"] == 2:
            raise RuntimeError("interrupted stage")

    monkeypatch.setattr(stages, "atomic_state", interrupted)
    with pytest.raises(RuntimeError, match="interrupted stage"):
        run(tmp_path / "resumed.pth")
    monkeypatch.setattr(stages, "atomic_state", save)
    actual = run(tmp_path / "resumed.pth")
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)


def test_runtime_state_contract_rejects_partial_hooks():
    from moderntsf.benchmark.infra.checkpoint import runtime_state, restore_runtime_state

    model = Forecaster()
    model.runtime_state_dict = lambda: {"cursor": 4}
    with pytest.raises(ValueError, match="both"):
        runtime_state(model)
    model.load_runtime_state_dict = lambda state: setattr(
        model, "cursor", state["cursor"]
    )
    restore_runtime_state(model, runtime_state(model))
    assert model.cursor == 4


def test_local_queue_recovers_after_controller_crash(tmp_path):
    import os
    import signal
    import time
    from moderntsf.benchmark.infra.queue import enqueue, work, jobs
    from moderntsf.benchmark.infra.execution import prepare_sweep
    from moderntsf.benchmark.config.loader import load_config

    loaded = load_config("configs/runs/smoke_crib.toml")
    loaded[0].config.experiment.work_dir = str(tmp_path / "outputs")
    sweep = prepare_sweep(loaded, ExecutionPolicy())
    queue = tmp_path / "queue"
    enqueue(queue, sweep)
    work(queue, once=True)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state = jobs(queue)[0]
        if state["status"] == "running":
            break
        time.sleep(0.05)
    assert state["status"] == "running"
    os.kill(state["pid"], signal.SIGKILL)
    time.sleep(0.3)
    work(queue, once=True)
    while time.monotonic() < deadline:
        state = jobs(queue)[0]
        if state["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.1)
    assert state["status"] == "succeeded", state


def test_evaluation_resumes_predictions_and_rng(tmp_path, monkeypatch):
    from moderntsf.benchmark.runner.evaluator import evaluate
    import moderntsf.benchmark.infra.stages as stages

    def run(checkpoint):
        torch.manual_seed(43)
        model = Forecaster()
        loader = DataLoader(
            TensorDataset(
                torch.ones(8, 4, 1),
                torch.ones(8, 2, 1),
                torch.zeros(8, 4, 1),
                torch.zeros(8, 2, 1),
            ),
            batch_size=2,
        )
        return evaluate(
            model,
            loader,
            torch.device("cpu"),
            0,
            2,
            "M",
            checkpoint=checkpoint,
            checkpoint_every_batches=1,
        )[0]

    expected = run(None)
    save = stages.atomic_state

    def interrupted(path, state):
        save(path, state)
        if state["next_batch"] == 2:
            raise RuntimeError("evaluation interrupted")

    monkeypatch.setattr(stages, "atomic_state", interrupted)
    with pytest.raises(RuntimeError, match="evaluation interrupted"):
        run(tmp_path / "evaluation.pth")
    monkeypatch.setattr(stages, "atomic_state", save)
    actual = run(tmp_path / "evaluation.pth")
    import numpy as np

    np.testing.assert_allclose(
        list(actual.values()), list(expected.values()), rtol=0, atol=0, equal_nan=True
    )

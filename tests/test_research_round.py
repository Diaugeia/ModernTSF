"""Tests for lightweight research rounds and Agent task startup."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tsflab.benchmark.research_round import (
    ResearchRoundError,
    add_event,
    claim_run,
    create_round,
    finish_run,
    events_for_run,
    list_rounds,
    load_round,
    read_events,
    set_status,
    write_log,
    write_prompt,
)


@pytest.fixture(autouse=True)
def isolated_work_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TSF_WORK_DIR", str(tmp_path / "work"))


def test_round_lifecycle_preserves_small_structured_memory() -> None:
    state = create_round(task="experiment", goal="Compare A against B", max_runs=2)
    round_id = state["id"]
    assert load_round(round_id)["runs_used"] == 0
    assert list_rounds()[0]["id"] == round_id

    add_event(round_id, "decision", "Use one representative horizon")
    first = claim_run(round_id, {"model": "Linear", "dataset": "synthetic"})
    finish_run(round_id, first, status="passed", run_id="run-one", metrics={"mse": 1.0})
    second = claim_run(round_id, {"model": "DLinear", "dataset": "synthetic"})
    finish_run(round_id, second, status="failed", error="fixture failure")

    with pytest.raises(ResearchRoundError, match="exhausted"):
        claim_run(round_id, {"model": "NLinear"})

    prompt = write_prompt(round_id, "Run the bounded comparison.\n")
    log = write_log(round_id, "config/unsafe", "complete output")
    second_log = write_log(round_id, "config/unsafe", "second output")
    assert prompt.read_text() == "Run the bounded comparison.\n"
    assert log.name == "config_unsafe.log"
    assert "complete output" in log.read_text()
    assert second_log != log
    assert second_log.read_text() == "second output"

    completed = set_status(round_id, "completed", "Evidence supports A")
    assert completed["status"] == "completed"
    events = read_events(round_id)
    assert {event["kind"] for event in events} >= {
        "hypothesis",
        "decision",
        "run",
        "failure",
        "conclusion",
    }
    assert not any("schema" in event for event in events)
    exported = events_for_run("run-one")
    assert len(exported) == len(events)
    assert all(event["round"] == round_id for event in exported)


def test_agent_task_start_writes_a_directly_readable_prompt(capsys) -> None:
    from tsflab.benchmark.commands.agent_tasks import agent_command

    code = agent_command(
        [
            "task",
            "start",
            "experiment",
            "--set",
            "question=Does normalization improve MSE?",
            "--set",
            "max_runs=2",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    round_id = payload["round"]["id"]
    assert payload["round"]["max_runs"] == 2
    assert payload["dispatch"] == "not-performed"
    assert payload["task"]["task"] == "experiment"
    assert "Does normalization improve MSE?" in payload["task"]["prompt"]
    assert load_round(round_id)["task"] == "experiment"


def test_sweep_association_counts_resolved_runs(monkeypatch) -> None:
    import importlib

    sweep_module = importlib.import_module("tsflab.benchmark.runner.run_sweep")
    state = create_round(task="experiment", goal="Count resolved runs", max_runs=1)
    monkeypatch.setenv("TSFLAB_RESEARCH_ROUND", state["id"])
    loaded = SimpleNamespace(
        config_name="fixture",
        config=SimpleNamespace(
            model=SimpleNamespace(name="Linear"),
            dataset=SimpleNamespace(name="synthetic"),
            experiment=SimpleNamespace(random_seed=0),
            task=SimpleNamespace(pred_len=4),
        ),
        raw={},
        sweep_keys=[],
    )
    result = SimpleNamespace(run_id="run-1", metrics={"mse": 0.5})
    monkeypatch.setattr(sweep_module, "run_one", lambda *args: result)

    assert sweep_module.run_sweep([loaded]) == [result]
    assert load_round(state["id"])["runs_used"] == 1
    assert any(
        event.get("details", {}).get("run_id") == "run-1"
        for event in read_events(state["id"])
    )

    with pytest.raises(ResearchRoundError, match="exhausted"):
        sweep_module.run_sweep([loaded])

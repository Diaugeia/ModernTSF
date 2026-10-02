"""Slot registry, Composed model, compose flags, and the result board (fast, no training)."""

from __future__ import annotations

import json
import py_compile
import tomllib
from pathlib import Path

import pytest

from tsflab.research.composition import compose_command, validate_composition
from tsflab.cli.commands.result_board import board
from tsflab.models._slots.registry import OPTIONS, check_assignment, resolve_assignment

ROOT = Path(__file__).resolve().parents[1]


def _spec(**slots):
    base = {
        "normalization": ["component:revin"],
        "decomposition": ["component:series_decomposition"],
        "temporal": ["component:channel_wise_linear"],
        "loss": ["loss:mae"],
    }
    base.update(slots)
    return {
        "name": "SeasonalRevLinear",
        "summary": "RevIN plus decomposition around a linear map.",
        "hypothesis": "seasonal split beats RLinear.",
        "parents": ["RLinear"],
        "slots": base,
    }


def _assignment(**over):
    a = {"normalization": "revin", "decomposition": "none", "temporal": "linear", "channel": "independent",
         "head": "flatten_forecast_head", "loss": "mse"}
    a.update(over)
    return a


def test_every_adapter_component_is_cataloged():
    from tsflab.catalog.components import COMPONENT_CATALOG

    for options in OPTIONS.values():
        for option in options.values():
            assert set(option.components) <= set(COMPONENT_CATALOG.names()), option


@pytest.mark.parametrize(
    "over, fragment",
    [
        ({"decomposition": "wavelet"}, "not executable"),
        ({"temporal": "mixer_block"}, "set channel to mixing"),
        ({"channel": "mixing"}, "only provided by mixer_block"),
        ({"head": "gaussian_parameter_head", "loss": "nll_gaussian", "channel": "individual"}, "shared weights only"),
        ({"head": "quantile_head"}, "needs loss 'quantile'"),
        ({"loss": "quantile"}, "needs a quantile_head"),
    ],
)
def test_unsupported_combinations_fail_with_a_reason(over, fragment):
    problems = check_assignment(_assignment(**over))
    assert any(fragment in p for p in problems), problems


def test_resolution_accepts_aliases_defaults_and_rejects_sweeps():
    resolved, errors = resolve_assignment(
        {"temporal": ["component:patchtst"], "normalization": ["component:revin"], "channel": ["local:individual"]}
    )
    assert not errors and resolved["temporal"] == "tst_transformer" and resolved["head"] == "flatten_forecast_head"
    _, errors = resolve_assignment({"temporal": ["component:mamba", "component:mixer_block"]})
    assert any("exactly one option" in e for e in errors)


def test_compose_reports_executability():
    result = validate_composition(_spec(), ROOT)
    assert result["ok"] and result["executable"]["ok"], result
    assert result["executable"]["assignment"]["loss"] == "mae"
    bad = validate_composition(_spec(temporal=["component:mixer_block"]), ROOT)
    assert bad["ok"] and not bad["executable"]["ok"]


def test_write_config_is_a_runnable_config(tmp_path):
    spec_path = tmp_path / "spec.toml"
    spec_path.write_text(
        'name = "SeasonalRevLinear"\nsummary = "s"\nhypothesis = "h"\nparents = ["RLinear"]\n'
        '[slots]\nnormalization = ["component:revin"]\ntemporal = ["component:channel_wise_linear"]\nloss = ["loss:mae"]\n'
        "[params]\nhidden = 8\n",
        encoding="utf-8",
    )
    out = tmp_path / "cfg" / "run.toml"
    rc = compose_command(
        [str(spec_path), "--write-config", str(out), "--dataset", "configs/fixtures/smoke.toml", "--enc-in", "6",
         "--pred-len", "12", "--smoke", "--work-dir", str(tmp_path / "work")],
        ROOT,
    )
    assert rc == 0
    cfg = tomllib.loads(out.read_text())
    assert [p.split("/")[-1] for p in cfg["extends"]] == ["base.toml", "smoke.toml", "Composed.toml"]
    assert all((out.parent / p).resolve().is_file() for p in cfg["extends"])
    assert cfg["model"]["name"] == "Composed"
    assert cfg["model"]["params"]["temporal"] == "channel_wise_linear" and cfg["model"]["params"]["hidden"] == 8
    assert cfg["training"]["loss"] == "mae" and cfg["task"]["pred_len"] == 12
    params = cfg["model"]["params"]
    from tsflab.catalog.registry.models import MODEL_CATALOG

    assert MODEL_CATALOG.get("Composed").validate_params(params)["enc_in"] == 6


def test_write_config_refuses_probabilistic_head_without_register(tmp_path):
    spec_path = tmp_path / "s.toml"
    spec_path.write_text(
        'name = "QRev"\nsummary = "s"\nhypothesis = "h"\nparents = ["RLinear"]\n[slots]\n'
        'temporal = ["component:channel_wise_linear"]\nhead = ["component:quantile_head"]\nloss = ["loss:quantile"]\n',
        encoding="utf-8",
    )
    rc = compose_command([str(spec_path), "--write-config", str(tmp_path / "r.toml"), "--dataset",
                          "configs/fixtures/smoke.toml", "--enc-in", "6"], ROOT)
    assert rc == 1 and not (tmp_path / "r.toml").exists()


def test_register_scaffolds_a_package_and_dry_run_writes_nothing(tmp_path):
    spec_path = tmp_path / "spec.toml"
    spec_path.write_text(
        'name = "QRev"\nsummary = "quantile head on a linear map"\nhypothesis = "calibrated intervals"\n'
        'parents = ["RLinear"]\n[slots]\nnormalization = ["component:revin"]\ntemporal = ["component:channel_wise_linear"]\n'
        'head = ["component:quantile_head"]\nloss = ["loss:quantile"]\n',
        encoding="utf-8",
    )
    root = tmp_path / "repo"
    assert compose_command([str(spec_path), "--register", "QRevLinear", "--dry-run", "--root", str(root)], ROOT) == 0
    assert not root.exists()
    assert compose_command([str(spec_path), "--register", "QRevLinear", "--root", str(root)], ROOT) == 0
    package = root / "src/tsflab/models/qrevlinear"
    for name in ("__init__.py", "model.py", "spec.py"):
        py_compile.compile(str(package / name), doraise=True)
    card = (package / "README.md").read_text()
    assert 'composition: ["normalization=component:revin"' in card and "quantile_head" in card
    assert "quantile-output" in (package / "spec.py").read_text()
    assert "[models.QRevLinear]" in (root / "verification/models.toml").read_text()
    assert (root / "configs/models/QRevLinear.toml").is_file()
    from tsflab.catalog.cards.models import _composition_problems

    fields = dict(line.split(": ", 1) for line in card.split("---")[1].strip().splitlines())
    assert not _composition_problems(json.loads(fields["composition"]), "QRev")


def test_board_merges_committed_rows_with_local_records(tmp_path):
    board_file = tmp_path / "leaderboard.json"
    board_file.write_text(json.dumps({
        "generated_at": "2026-01-01",
        "tracks": {"time_series": {"datasets": {"Toy": {"horizons": {"12": [
            {"model": "A", "mse": 0.5, "mae": 0.4, "n_runs": 1}, {"model": "B", "mse": 0.3, "mae": 0.3, "n_runs": 1}]}}}}},
    }))
    records = tmp_path / "work" / "Toy" / "Composed" / "records"
    records.mkdir(parents=True)
    (records / "r1.json").write_text(json.dumps({
        "record_id": "r1", "model": "Composed", "dataset_id": "Toy", "track": "time_series",
        "results": [{"horizon": 12, "metrics": {"mse": 0.2, "mae": 0.2}}],
        "config": {"seq_len": 96, "snapshot": {"training": {"epochs": 1}, "model": {"params": {"temporal": "linear", "normalization": "revin"}}}},
    }))
    result = board("toy", horizon="12", top=3, metric="mse", records=[tmp_path / "work"], board_file=str(board_file))
    rows = result["horizons"][0]["rows"]
    assert [r["model"] for r in rows] == ["Composed[revin/-/linear/-]", "B", "A"]
    assert rows[0]["source"] == "local" and rows[0]["epochs"] == 1 and rows[1]["source"] == "board"


def test_composed_model_runs_every_point_slot_combination():
    torch = pytest.importorskip("torch")
    from tsflab.models.composed.model import Model

    x = torch.randn(2, 48, 3)
    for temporal in ("linear", "channel_wise_linear", "tst_transformer", "mamba", "gated_dilated_conv"):
        for norm in ("none", "revin", "last_value_center"):
            for decomposition in ("none", "series_decomposition"):
                y = Model(3, 48, 12, normalization=norm, decomposition=decomposition, temporal=temporal,
                          patch_len=8, stride=4)(x)
                assert y.shape == (2, 12, 3) and torch.isfinite(y).all()
    y = Model(3, 48, 12, temporal="mixer_block", channel="mixing", normalization="revin")(x)
    assert y.shape == (2, 12, 3)
    with pytest.raises(ValueError, match="mixer_block"):
        Model(3, 48, 12, temporal="mixer_block")
    with pytest.raises(ValueError, match="point model"):
        Model(3, 48, 12, head="quantile_head")


def test_probabilistic_pipelines_emit_their_output_kind():
    torch = pytest.importorskip("torch")
    from tsflab.models._slots.adapters import Pipeline

    x = torch.randn(2, 48, 3)
    q = Pipeline(_assignment(head="quantile_head", loss="quantile", normalization="last_value_center"), 3, 48, 12)(x)
    assert q.shape == (2, 12, 3, 9) and bool((q.diff(dim=-1) >= 0).all())
    g = Pipeline(_assignment(head="gaussian_parameter_head", loss="nll_gaussian"), 3, 48, 12)(x)
    assert g.shape == (2, 12, 3, 2) and bool((g[..., 1] > 0).all())


def test_params_schema_rejects_invalid_slot_combinations():
    from pydantic import ValidationError

    from tsflab.catalog.registry.models import MODEL_CATALOG

    spec = MODEL_CATALOG.get("Composed")
    with pytest.raises(ValidationError, match="mixer_block"):
        spec.validate_params({"enc_in": 3, "temporal": "mixer_block"})
    with pytest.raises(ValidationError, match="not executable"):
        spec.validate_params({"enc_in": 3, "decomposition": "wavelet"})

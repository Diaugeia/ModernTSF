"""Dataset profile (`tsf dataset analyze`) and recombination-spec validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from moderntsf.benchmark.catalog.components import COMPONENT_CATALOG
from moderntsf.benchmark.catalog.composition import validate_composition
from moderntsf.benchmark.catalog_metadata import model_records
from moderntsf.benchmark.commands import dataset_analyze
from moderntsf.benchmark.registry.losses import LOSS_NAME_MAP
from moderntsf.data import profile as prof

ROOT = Path(__file__).resolve().parents[1]
PERIOD = 24


def _series(n=2400, period=PERIOD, slope=0.0, season=1.0, noise=0.3, channels=3, seed=0, shared=False):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = rng.normal(size=(n, channels)) * noise
    wave = season * np.sin(2 * np.pi * t / period)[:, None]
    out = base + slope * t[:, None] / n * 10 + wave
    if shared:
        out = out + 3 * rng.normal(size=(n, 1)).cumsum(axis=0) / 30
    return out


def _split(x, ratio=(0.7, 0.1, 0.2)):
    a, b = int(ratio[0] * len(x)), int((ratio[0] + ratio[1]) * len(x))
    return {"train": x[:a], "val": x[a:b], "test": x[b:]}


def _profile(x, **kw):
    return prof.build_profile(_split(x), name="syn", frequency=prof.infer_frequency(None), **kw)


def test_recovers_period_and_seasonal_strength():
    p = _profile(_series())
    top = p["train"]["periods"]["dominant"][0]
    assert abs(top["period"] - PERIOD) <= 1
    assert p["train"]["seasonal"]["strength"] > 0.8
    assert p["train"]["periods"]["forecastability"]["score"] > 0.5
    assert any(c["value"] % PERIOD == 0 for c in p["lookback_candidates"])


def test_trend_strength_separates_trend_from_flat():
    trended = _profile(_series(slope=1.0, season=0.2, noise=0.1))
    flat = _profile(_series(slope=0.0, season=1.0, noise=0.3))
    assert trended["train"]["trend"]["strength"] > 0.8
    assert flat["train"]["trend"]["strength"] < trended["train"]["trend"]["strength"]
    assert trended["train"]["stationarity"]["mean_drift"] > flat["train"]["stationarity"]["mean_drift"]


def test_white_noise_has_no_structure():
    p = _profile(np.random.default_rng(1).normal(size=(2400, 3)))
    assert p["train"]["seasonal"]["strength"] < 0.3
    assert p["train"]["periods"]["forecastability"]["score"] < 0.2
    assert "strong-seasonality" not in {r["id"] for r in prof.evaluate_rules(p)["fired"]}


def test_cross_channel_structure():
    rng = np.random.default_rng(2)
    common = rng.normal(size=(2400, 1)).cumsum(axis=0)
    correlated = common + 0.05 * rng.normal(size=(2400, 6))
    independent = rng.normal(size=(2400, 6))
    hi = _profile(correlated)["train"]["cross_channel"]
    lo = _profile(independent)["train"]["cross_channel"]
    assert hi["mean_abs_corr"] > 0.9 and lo["mean_abs_corr"] < 0.2
    assert hi["pc1_share"] > lo["pc1_share"]
    assert _profile(_series(channels=1))["train"]["cross_channel"] == {"applicable": False}


def test_zero_inflation_missingness_and_outliers():
    x = _series(channels=2)
    x[::2] = 0.0
    x[5, 0] = np.nan
    x[100, 1] = 500.0
    q = _profile(x)["train"]["quality"]
    assert q["zero_fraction"] > 0.4 and q["missing_fraction"] > 0
    assert _profile(x)["train"]["anomalies"]["outlier_fraction_5mad"] is not None


def test_shift_is_split_aware_and_test_is_diagnostic_only():
    x = _series()
    splits = _split(x)
    shifted_val = {**splits, "val": splits["val"] + 4.0}
    p = prof.build_profile(shifted_val, name="s")
    assert p["shift"]["train_to_val"]["mean_shift_median"] > 2.0
    assert "diagnostic-only" in p["shift"]["train_to_test"]["use"]
    assert prof.lookup(p, "shift.train_to_test.mean_shift_median") is not None


def test_changing_test_split_never_changes_recommendations_or_lookbacks():
    splits = _split(_series(seed=3))
    base = prof.build_profile(splits, name="a")
    tampered = prof.build_profile({**splits, "test": splits["test"] * 50 + 100}, name="a")
    assert prof.evaluate_rules(base) == prof.evaluate_rules(tampered)
    assert base["lookback_candidates"] == tampered["lookback_candidates"]
    assert base["train"] == tampered["train"]


def test_lookbacks_respect_capacity():
    p = _profile(_series(n=400))
    assert max(c["value"] for c in p["lookback_candidates"]) <= 400 * 0.7 // 4


def test_too_short_train_is_rejected():
    with pytest.raises(ValueError):
        prof.build_profile({"train": np.zeros((10, 2)), "val": np.zeros((4, 2)), "test": np.zeros((4, 2))}, name="x")


def test_frequency_inference_and_period_tags():
    import pandas as pd

    f = prof.infer_frequency(pd.date_range("2020-01-01", periods=50, freq="10min"))
    assert f["label"] == "10min" and f["steps_per_day"] == 144.0
    assert prof._period_tag(143.7, 144.0) == "daily"
    assert prof._period_tag(1008.0, 144.0) == "weekly"


# --------------------------------------------------------------------------- #
# Rules table is data, and names in it must exist in the catalog
# --------------------------------------------------------------------------- #
def _catalog_names():
    return {str(r["name"]) for r in model_records(ROOT)}


def test_rule_table_names_exist_and_metrics_are_decision_safe():
    table = prof.load_rules()
    components, models = set(COMPONENT_CATALOG.names()), _catalog_names()
    slots = set(table["slot"])
    assert len({r["id"] for r in table["rule"]}) == len(table["rule"])

    def check(option: str, where: str):
        kind, _, value = option.partition(":")
        known = {"component": components, "model": models, "loss": set(LOSS_NAME_MAP)}
        assert kind in known and value in known[kind], f"{where}: {option}"

    for name, spec in table["slot"].items():
        for option in spec["defaults"]:
            check(option, f"slot {name}")
    for rule in table["rule"]:
        assert rule["metric"].startswith(prof.DECISION_PREFIXES), rule["id"]
        assert "train_to_test" not in rule["metric"], rule["id"]
        assert rule["op"] in prof._OPS
        assert set(rule["components"]) <= components, rule["id"]
        assert set(rule["models"]) <= models, rule["id"]
        for slot, options in rule.get("slots", {}).items():
            assert slot in slots, rule["id"]
            for option in options:
                check(option, rule["id"])


def test_every_rule_metric_resolves_in_a_real_profile():
    p = _profile(_series())
    for rule in prof.load_rules()["rule"]:
        assert prof.lookup(p, rule["metric"]) is not None, rule["metric"]


def test_seasonal_series_fires_seasonality_rule_and_promotes_slots():
    p = _profile(_series())
    result = prof.evaluate_rules(p)
    assert "strong-seasonality" in {r["id"] for r in result["fired"]}
    promoted = {o["option"] for o in result["slots"]["decomposition"]["promoted"]}
    assert "component:series_decomposition" in promoted
    assert "# Dataset profile" in prof.render_markdown({**p, "recommendations": result}, result)


# --------------------------------------------------------------------------- #
# CLI on a raw file
# --------------------------------------------------------------------------- #
def test_analyze_cli_on_csv_writes_json_and_markdown(tmp_path, capsys):
    import pandas as pd

    x = _series(n=1500, channels=2)
    frame = pd.DataFrame(x, columns=["a", "b"])
    frame.insert(0, "date", pd.date_range("2021-01-01", periods=len(x), freq="h"))
    path = tmp_path / "syn.csv"
    frame.to_csv(path, index=False)
    out = tmp_path / "out"
    assert dataset_analyze.main(["--path", str(path), "--out", str(out), "--json"]) == 0
    report = json.loads((out / "profile.json").read_text())
    assert report["dataset"]["frequency"]["label"] == "1h"
    assert abs(report["train"]["periods"]["dominant"][0]["period"] - PERIOD) <= 1
    assert report["train"]["periods"]["dominant"][0]["tag"] == "daily"
    assert (out / "profile.md").read_text().startswith("# Dataset profile")
    with pytest.raises(SystemExit):
        dataset_analyze.main([str(path), "--path", str(path)])


# --------------------------------------------------------------------------- #
# Composition spec validation
# --------------------------------------------------------------------------- #
def _spec(**over):
    spec = {
        "name": "SeasonalRevLinear",
        "summary": "RevIN plus decomposition around a linear map.",
        "hypothesis": "strong-seasonality: period-aware split beats RLinear.",
        "parents": ["RLinear"],
        "slots": {
            "normalization": ["component:revin"],
            "decomposition": ["component:series_decomposition"],
            "temporal": ["component:channel_wise_linear"],
            "loss": ["loss:mae"],
        },
    }
    spec.update(over)
    return spec


def test_valid_composition_yields_scaffold_and_provenance():
    result = validate_composition(_spec(), ROOT)
    assert result["ok"], result["errors"]
    assert result["components"] == ["revin", "series_decomposition", "channel_wise_linear"]
    assert "--components revin,series_decomposition,channel_wise_linear" in result["scaffold"]
    assert "RLinear" in result["card_provenance"]


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda s: s["slots"].update(temporal=["component:no_such_block"]), "unknown component"),
        (lambda s: s["slots"].update(loss=["loss:mae", "loss:mse"]), "one training loss"),
        (lambda s: s["slots"].update(channel=["model:NoSuchModel"]), "unknown donor"),
        (lambda s: s["slots"].update(bogus=["component:revin"]), "unknown slot"),
        (lambda s: s.update(name="RLinear"), "already exists"),
        (lambda s: s.update(parents=["Nope"]), "not in the catalog"),
        (lambda s: s["slots"].update(head=["revin"]), "prefix"),
        (lambda s: s.update(free_form=[{"slot": "temporal", "name": "A", "description": "d", "max_lines": 500}]), "max_lines"),
        (lambda s: s["slots"].pop("temporal"), "temporal slot"),
    ],
)
def test_invalid_compositions_are_rejected(mutate, fragment):
    spec = _spec()
    mutate(spec)
    result = validate_composition(spec, ROOT)
    assert not result["ok"]
    assert any(fragment in e for e in result["errors"]), result["errors"]


def test_free_form_block_fills_a_slot_within_budget():
    spec = _spec(free_form=[{"slot": "temporal", "name": "PhaseGate", "description": "gate", "max_lines": 60}])
    del spec["slots"]["temporal"]
    result = validate_composition(spec, ROOT)
    assert result["ok"], result["errors"]
    assert result["free_form"] == ["PhaseGate"]

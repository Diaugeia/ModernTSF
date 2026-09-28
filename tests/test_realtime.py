"""Rolling real-time evaluation: store, rounds, submissions, scoring, parsers."""

from __future__ import annotations

from dataclasses import replace
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moderntsf.realtime import rounds as R
from moderntsf.realtime.baselines import run_baselines
from moderntsf.realtime.sources.openaq import parse_hours
from moderntsf.realtime.sources.pems import parse_station_5min
from moderntsf.realtime.store import PanelStore
from moderntsf.realtime.tracks import TrackSpec, get_track, list_tracks
from moderntsf.tsf_core.realtime import ForecastSubmission

TRACK = TrackSpec(id="toy", title="toy", mode="time_series", freq="h", seq_len=24, horizon=6,
                  submission_hours=2, seasonal_period=24, min_coverage=0.8, tz="America/Los_Angeles")


def _panel(start: str, hours: int, channels: int = 3) -> pd.DataFrame:
    index = pd.date_range(start, periods=hours, freq="h")
    t = np.arange(hours)[:, None]
    values = 100 + 10 * np.sin(2 * np.pi * t / 24) + np.arange(channels)[None, :]
    return pd.DataFrame(values, index=index, columns=[f"s{i}" for i in range(channels)])


def test_shipped_track_configs_load() -> None:
    ids = {track.id for track in list_tracks()}
    assert {"stock_hs300", "traffic_pems_sb", "air_openaq_cn"} <= ids
    assert get_track("traffic_pems_sb").tz == "America/Los_Angeles"


def test_store_appends_without_overwriting_observed_cells(tmp_path: Path) -> None:
    store = PanelStore("toy", tmp_path)
    store.append(_panel("2026-01-01", 48))
    revised = _panel("2026-01-02", 48) + 1000  # overlaps day 2 with different values
    revised["new_sensor"] = 1.0
    store.append(revised)
    panel = store.read()
    assert list(panel.columns) == ["s0", "s1", "s2"]  # channel set stays fixed
    assert panel.loc["2026-01-02 05:00", "s0"] < 200  # history was not rewritten
    assert panel.loc["2026-01-03 05:00", "s0"] > 1000  # new timestamps were added
    assert len(store.manifest()["releases"]) == 2


def _opened_round(tmp_path: Path):
    store = PanelStore("toy", tmp_path / "store")
    store.append(_panel("2026-01-01", 24 * 10))
    cutoff = store.read().index.max()
    opened = (cutoff + pd.Timedelta(hours=1)).tz_localize(TRACK.tz)
    spec = R.open_round(TRACK, store, now=opened.to_pydatetime(), root=tmp_path / "rounds")
    return store, spec


def test_round_targets_follow_the_submission_window(tmp_path: Path) -> None:
    store, spec = _opened_round(tmp_path)
    assert spec.horizon == 6
    first = pd.Timestamp(spec.target_timestamps[0])
    assert first - pd.Timestamp(spec.opened_at) >= pd.Timedelta(hours=TRACK.submission_hours)
    assert pd.Timestamp(spec.deadline) < first
    assert first.tzinfo is not None  # every stored instant carries its offset


def test_late_or_malformed_forecasts_are_rejected(tmp_path: Path) -> None:
    _, spec = _opened_round(tmp_path)
    good = ForecastSubmission(track="toy", round_id=spec.round_id, model="M",
                              submitted_at=spec.opened_at, predictions=[[0.0] * 3] * spec.horizon)
    good.check_against(spec)
    late = good.model_copy(update={"submitted_at": spec.target_timestamps[0]})
    with pytest.raises(ValueError, match="deadline"):
        late.check_against(spec)
    with pytest.raises(ValueError, match="forecast steps"):
        good.model_copy(update={"predictions": [[0.0] * 3]}).check_against(spec)
    with pytest.raises(ValueError, match="finite"):
        good.model_copy(update={"predictions": [[float("nan")] * 3] * spec.horizon}).check_against(spec)


def test_scoring_waits_for_truth_then_ranks(tmp_path: Path) -> None:
    store, spec = _opened_round(tmp_path)
    root = tmp_path / "rounds"
    for submission in run_baselines(spec, store, TRACK):
        R.write_forecast(submission.model_copy(update={"submitted_at": spec.opened_at}), spec, root)
    assert R.score_round(spec, store, TRACK.min_coverage, root) is None  # no truth yet
    store.append(_panel("2026-01-11", 48))
    scores = R.score_round(spec, store, TRACK.min_coverage, root)
    assert [s.rank for s in scores] == [1, 2, 3]
    assert scores[0].model == "SeasonalNaive"  # exact daily seasonality in the toy panel
    assert scores[0].mse < 1e-8
    summary = R.track_summary("toy", root)
    assert summary["methods"][0]["model"] == "SeasonalNaive"


def test_kendall_tau() -> None:
    assert R.kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert R.kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    assert R.kendall_tau(["a"], ["a"]) is None


def test_pems_station_5min_parser_aggregates_hourly_flow() -> None:
    rows = []
    for minute in range(0, 60, 5):
        for station, flow in (("801230", 10), ("801232", 20)):
            rows.append(f"01/05/2026 07:{minute:02d}:00,{station},8,10,E,ML,1.2,10,100,{flow},0.05,65")
    payload = gzip.compress("\n".join(rows).encode())
    hourly = parse_station_5min(payload)
    assert hourly.loc[pd.Timestamp("2026-01-05 07:00"), "801230"] == 120
    assert hourly.loc[pd.Timestamp("2026-01-05 07:00"), "801232"] == 240


def test_openaq_hours_parser() -> None:
    rows = [{"value": 12.5, "period": {"datetimeFrom": {"utc": "2026-01-05T07:00:00Z"}}},
            {"value": None, "period": {"datetimeFrom": {"utc": "2026-01-05T08:00:00Z"}}}]
    series = parse_hours(rows)
    assert list(series.values) == [12.5]
    assert series.index[0] == pd.Timestamp("2026-01-05 07:00")


def test_validate_cli_uses_trusted_arrival_time(tmp_path: Path, monkeypatch) -> None:
    from moderntsf.realtime.cli import main

    _, spec = _opened_round(tmp_path)
    monkeypatch.setattr(R, "ROUNDS_ROOT", tmp_path / "rounds")
    monkeypatch.setattr(R, "load_round", lambda t, r, root=tmp_path / "rounds": R.RoundSpec.model_validate_json(
        (tmp_path / "rounds" / t / "rounds" / r / "round.json").read_text()))
    path = tmp_path / "f.json"
    path.write_text(ForecastSubmission(track="toy", round_id=spec.round_id, model="M",
                                       submitted_at=spec.opened_at,
                                       predictions=[[0.0] * 3] * spec.horizon).model_dump_json())
    assert main(["validate", str(path)]) == 0
    assert main(["validate", str(path), "--received-at", spec.target_timestamps[-1]]) == 1

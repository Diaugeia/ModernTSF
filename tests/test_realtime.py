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
    assert {"stock_hs300", "stock_nasdaq100", "traffic_pems_sb", "air_openaq_cn"} <= ids
    assert get_track("stock_nasdaq100").source["kind"] == "nasdaq100"
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


def test_equity_fallback_moves_the_working_vendor_first(monkeypatch) -> None:
    from moderntsf.realtime.sources import equity

    monkeypatch.setattr(equity.time, "sleep", lambda seconds: None)
    calls = []

    def blocked(symbol):
        calls.append("blocked")
        raise ConnectionError("403")

    def working(symbol):
        calls.append("working")
        return pd.Series([1.0])

    providers = [blocked, working]
    assert equity.with_fallback(providers, "AAPL").tolist() == [1.0]
    assert providers == [working, blocked]
    equity.with_fallback(providers, "MSFT")
    assert calls == ["blocked", "working", "working"]
    with pytest.raises(RuntimeError, match="could not fetch X"):
        equity.with_fallback([blocked], "X", attempts=2, label="X")


def test_equity_panel_stores_log_returns_and_resumes_from_cache(tmp_path: Path, monkeypatch) -> None:
    from moderntsf.realtime.sources import equity

    monkeypatch.setenv("MODERNTSF_REALTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(equity.time, "sleep", lambda seconds: None)
    track = replace(TRACK, id="toy_stock", freq="B", source={"transform": "log_return"})
    days = pd.bdate_range("2026-08-24", "2026-09-11")
    closes = pd.Series(100 * np.exp(0.01 * np.arange(len(days))), index=days)
    fetched = []

    def daily_close(symbol, lo, hi):
        fetched.append(symbol)
        return closes.loc[lo:hi]

    start, end = pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-11")
    panel = equity.fetch_panel(track, start, end, ["A", "B"], daily_close)
    assert list(panel.columns) == ["A", "B"]
    assert panel.index[0] == start and panel.index[-1] == end
    np.testing.assert_allclose(panel.to_numpy(), 0.01)  # first day uses the previous close
    again = equity.fetch_panel(track, start, end, ["A", "B"], daily_close)
    assert fetched == ["A", "B"]  # second call is served from the resume cache
    pd.testing.assert_frame_equal(again, panel, check_freq=False)


def test_nasdaq_historical_fallback_parses_quoted_closes(monkeypatch) -> None:
    from moderntsf.realtime.sources import nasdaq100, us_prices

    rows = [{"date": "09/25/2026", "close": "$1,341.07"}, {"date": "09/24/2026", "close": "$335.92"}]
    seen = {}

    def fake_get(path, **params):
        seen["path"] = path
        return {"tradesTable": {"rows": rows}}

    monkeypatch.setattr(us_prices, "nasdaq_get", fake_get)
    close = us_prices.nasdaq_close("BRK-B", pd.Timestamp("2026-09-24"), pd.Timestamp("2026-09-25"))
    assert seen["path"] == "quote/BRK.B/historical"
    assert close.tolist() == [335.92, 1341.07]
    assert nasdaq100._PREFERRED[:2] == [us_prices.nasdaq_close, us_prices.yahoo_close]  # Sina is last
    monkeypatch.setattr(nasdaq100, "nasdaq_get", lambda path, **params: {"data": {"rows": [
        {"symbol": "brk.b "}, {"symbol": "AAPL"}, {"symbol": "AAPL"}]}})
    assert nasdaq100.constituents() == ["AAPL", "BRK-B"]

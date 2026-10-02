"""Real-time source parsers and fetch plumbing on recorded responses (no network)."""

from __future__ import annotations

from dataclasses import replace
import io
import zipfile

import pandas as pd
import pytest

from tsflab.realtime.sources import airnow, eia930, ercot, openaq, openmeteo, sp500, us_prices
from tsflab.realtime.tracks import get_track, list_tracks

NEW_TRACKS = ["weather_openmeteo_temp", "solar_openmeteo_ghi", "air_airnow_us", "air_openaq_us",
              "air_openaq_eu", "grid_ercot", "grid_eia_us", "solar_eia_us", "stock_sp500"]


def test_new_track_configs_load_and_resolve_to_sources() -> None:
    ids = {t.id for t in list_tracks()}
    assert set(NEW_TRACKS) <= ids
    kinds = {"weather_openmeteo_temp": "openmeteo", "solar_openmeteo_ghi": "openmeteo", "air_airnow_us": "airnow",
             "air_openaq_us": "openaq", "air_openaq_eu": "openaq", "grid_ercot": "ercot",
             "grid_eia_us": "eia930", "solar_eia_us": "eia930", "stock_sp500": "sp500"}
    for track_id, kind in kinds.items():
        track = get_track(track_id)
        assert track.source["kind"] == kind and track.bootstrap["kind"] == "source"
        assert track.horizon >= 1 and track.seq_len >= 1


@pytest.mark.parametrize("track_id,low,high", [("weather_openmeteo_temp", 70, 100), ("solar_openmeteo_ghi", 40, 100)])
def test_open_meteo_site_lists_are_sane(track_id: str, low: int, high: int) -> None:
    sites = get_track(track_id).source["sites"]
    assert low <= len(sites) <= high
    assert len({s[0] for s in sites}) == len(sites)
    assert all(-90 <= s[1] <= 90 and -180 <= s[2] <= 180 for s in sites)


def test_open_meteo_parser_handles_batches_and_single_site() -> None:
    hourly = {"time": ["2026-09-20T00:00", "2026-09-20T01:00"], "temperature_2m": [10.5, None]}
    batch = openmeteo.parse_hourly([{"hourly": hourly}, {"hourly": {**hourly, "temperature_2m": [1.0, 2.0]}}],
                                   "temperature_2m", ["a", "b"])
    assert batch.loc["2026-09-20 00:00", "a"] == 10.5 and pd.isna(batch.loc["2026-09-20 01:00", "a"])
    assert list(batch["b"]) == [1.0, 2.0]
    assert list(openmeteo.parse_hourly({"hourly": hourly}, "temperature_2m", ["a"]).columns) == ["a"]
    with pytest.raises(ValueError):
        openmeteo.parse_hourly([{"hourly": hourly}], "temperature_2m", ["a", "b"])


def test_open_meteo_fetch_clips_recent_hours_and_batches(monkeypatch) -> None:
    track = get_track("weather_openmeteo_temp")
    track = replace(track, source={**track.source, "sites": track.source["sites"][:3], "batch": 2,
                                   "pause_seconds": 0, "lag_hours": 3})
    calls = []

    def fake(params):
        calls.append(params)
        n = len(params["latitude"].split(","))
        index = pd.date_range(params["start_date"], pd.Timestamp(params["end_date"]) + pd.Timedelta(hours=23), freq="h")
        return [{"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in index], "temperature_2m": [1.0] * len(index)}}
                for _ in range(n)]

    monkeypatch.setattr(openmeteo, "_get", fake)
    monkeypatch.setattr(openmeteo.time, "sleep", lambda s: None)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    panel = openmeteo.fetch(track, now - pd.Timedelta(days=2), now, None)
    assert len(calls) == 2  # three sites in batches of two
    assert panel.index.max() <= now - pd.Timedelta(hours=3)  # forecast hours are never stored
    assert list(panel.columns) == [s[0] for s in track.source["sites"][:3]]


AIRNOW = (
    "10/02/26|11:00|840010491003|Site A|-4|PM2.5|UG/M3|12.0|State A\r\n"
    "10/02/26|11:00|840010491003|Site A|-4|OZONE|PPB|30|State A\r\n"
    "10/02/26|11:00|000020104|CHARLOTTETOWN|-4|PM2.5|UG/M3|4.7|Canada\r\n"
    "10/02/26|11:00|840020900040|Site B|-9|PM2.5|UG/M3|-0.1|State B\r\n"
    "10/02/26|11:00|840021700010|Site C|-9|PM2.5|UG/M3||State C\r\n"
)


def test_airnow_parser_keeps_us_pm25_in_utc() -> None:
    series = airnow.parse_hourly_file(AIRNOW)
    assert len(series) == 2  # ozone, Canadian and empty rows are dropped
    assert series[(pd.Timestamp("2026-10-02 11:00"), "840010491003")] == 12.0
    assert series[(pd.Timestamp("2026-10-02 11:00"), "840020900040")] == -0.1  # raw instrument noise is kept
    assert airnow.hour_url(pd.Timestamp("2026-10-02 11:00")).endswith("/2026/20261002/HourlyData_2026100211.dat")


def test_airnow_fetch_selects_best_covered_sites_then_keeps_them(monkeypatch) -> None:
    track = replace(get_track("air_airnow_us"), source={**get_track("air_airnow_us").source,
                                                        "max_sites": 1, "min_coverage": 0.5, "workers": 1})

    def lines(hour, with_b):
        rows = [f"10/02/26|{hour:02d}:00|840010491003|A|-4|PM2.5|UG/M3|{hour}|x"]
        if with_b:
            rows.append(f"10/02/26|{hour:02d}:00|840020900040|B|-4|PM2.5|UG/M3|1|x")
        return "\n".join(rows)

    monkeypatch.setattr(airnow, "_download", lambda h: None if h.hour == 12 else lines(h.hour, h.hour == 10))
    panel = airnow.fetch(track, pd.Timestamp("2026-10-02 10:00"), pd.Timestamp("2026-10-02 12:00"), None)
    assert list(panel.columns) == ["840010491003"] and len(panel) == 2  # hour 12 was not published
    both = airnow.fetch(track, pd.Timestamp("2026-10-02 10:00"), pd.Timestamp("2026-10-02 11:00"),
                        ["840010491003", "840020900040"])
    assert list(both.columns) == ["840010491003", "840020900040"]


def _tiny_xlsx(rows: list[list[object]]) -> bytes:
    shared: list[str] = []

    def cell(ref: str, value: object) -> str:
        if isinstance(value, str):
            shared.append(value)
            return f'<c r="{ref}" t="s"><v>{len(shared) - 1}</v></c>'
        return f'<c r="{ref}"><v>{value}</v></c>'

    body = "".join(
        f'<row r="{i + 1}">' + "".join(cell(f"{chr(65 + j)}{i + 1}", v) for j, v in enumerate(row)) + "</row>"
        for i, row in enumerate(rows))
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as book:
        book.writestr("xl/worksheets/sheet1.xml", f"<worksheet {ns}><sheetData>{body}</sheetData></worksheet>")
        book.writestr("xl/sharedStrings.xml", f"<sst {ns}>" + "".join(f"<si><t>{s}</t></si>" for s in shared) + "</sst>")
    return out.getvalue()


def test_ercot_native_load_archive_is_read_without_openpyxl() -> None:
    header = ["Hour Ending", *ercot.ZONES, "ERCOT"]
    rows = [header,
            ["03/09/2025 01:00", *range(1, 9), 36],
            ["03/09/2025 24:00", *range(11, 19), 116],
            ["11/02/2025 01:00", *[100] * 8, 800],
            ["11/02/2025 02:00", *[100] * 8, 800],
            ["11/02/2025 02:00", *[300] * 8, 2400]]  # repeated hour of the clock change
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.writestr("Native_Load_2025.xlsx", _tiny_xlsx(rows))
    frame = ercot.parse_native_load(zipped.getvalue())
    assert list(frame.columns) == ercot.ZONES
    assert frame.loc["2025-03-09 00:00", "COAST"] == 1  # hour ending 01:00 covers 00:00-01:00
    assert frame.loc["2025-03-09 23:00", "COAST"] == 11  # 24:00 is the last hour of the same day
    assert frame.loc["2025-11-02 01:00", "COAST"] == 200  # duplicate hour averaged
    assert frame.index.is_unique


def test_ercot_daily_weather_zone_report_matches_archive_names() -> None:
    text = ("OperDay,HourEnding,COAST,EAST,FAR_WEST,NORTH,NORTH_C,SOUTHERN,SOUTH_C,WEST,TOTAL,DSTFlag\n"
            "10/01/2026,01:00,16023.5,2034.95,7333.67,2026.75,16451.46,5343.57,10647.45,1569.58,61430.93,N\n"
            "10/01/2026,02:00,15417.71,1925.37,7213.02,2191.10,15915.20,5158.56,10167.12,1309.80,59297.89,N\n")
    frame = ercot.parse_weather_zone_csv(text)
    assert list(frame.columns) == ercot.ZONES
    assert frame.index[0] == pd.Timestamp("2026-10-01 00:00")
    assert frame.loc["2026-10-01 00:00", "FWEST"] == 7333.67


def test_ercot_fetch_prefers_archive_and_fills_from_daily_reports(monkeypatch) -> None:
    track = get_track("grid_ercot")
    index = pd.date_range("2026-08-30", "2026-09-02 23:00", freq="h")
    archive = pd.DataFrame(1.0, index=index[index < "2026-09-01"], columns=ercot.ZONES)
    daily = pd.DataFrame(2.0, index=index[index >= "2026-08-31"], columns=ercot.ZONES)
    monkeypatch.setattr(ercot, "archive_urls", lambda: {2026: "u"})
    monkeypatch.setattr(ercot, "_get", lambda url, **kw: type("R", (), {"content": b""})())
    monkeypatch.setattr(ercot, "parse_native_load", lambda payload: archive)
    monkeypatch.setattr(ercot, "recent_daily_texts", lambda since: ["x"])
    monkeypatch.setattr(ercot, "parse_weather_zone_csv", lambda text: daily)
    panel = ercot.fetch(track, pd.Timestamp("2026-08-30"), pd.Timestamp("2026-09-02 23:00"), None)
    assert panel.loc["2026-08-31 12:00", "COAST"] == 1.0 and panel.loc["2026-09-01 12:00", "COAST"] == 2.0
    assert panel.index.is_unique and len(panel) == len(index)


def test_eia_rows_pivot_to_hourly_balancing_authorities() -> None:
    rows = [{"period": "2026-10-02T15", "respondent": "ERCO", "value": "58148"},
            {"period": "2026-10-02T14", "respondent": "ERCO", "value": "56638"},
            {"period": "2026-10-02T15", "respondent": "PJM", "value": "90000"},
            {"period": "2026-10-02T14", "respondent": "PJM", "value": None}]
    frame = eia930.parse_rows(rows)
    assert frame.loc["2026-10-02 15:00", "ERCO"] == 58148.0
    assert frame.shape == (2, 2) and pd.isna(frame.loc["2026-10-02 14:00", "PJM"])


def test_eia_fetch_paginates_drops_aggregates_and_needs_a_key(monkeypatch) -> None:
    track = get_track("grid_eia_us")
    with pytest.raises(RuntimeError, match="EIA_API_KEY"):
        eia930.fetch(track, pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-02"), None)
    pages = [[{"period": "2026-10-02T1%d" % h, "respondent": r, "value": "1"} for h in range(2) for r in ("ERCO", "US48")]]
    monkeypatch.setattr(eia930, "_get", lambda path, params: {"data": pages[0]})
    monkeypatch.setattr(eia930, "PAGE", 100)
    panel = eia930.fetch(track, pd.Timestamp("2026-10-02 10:00"), pd.Timestamp("2026-10-02 11:00"), None)
    assert list(panel.columns) == ["ERCO"]  # US48 is an interconnection aggregate


def test_nasdaq_history_is_sorted_oldest_first_and_empty_raises() -> None:
    data = {"tradesTable": {"rows": [{"date": "10/01/2026", "close": "$330.32"},
                                      {"date": "09/30/2026", "close": "$1,333.02"}]}}
    series = us_prices.parse_nasdaq_history(data)
    assert list(series.index) == [pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-01")]
    assert series.iloc[0] == 1333.02
    with pytest.raises(RuntimeError):
        us_prices.parse_nasdaq_history({"tradesTable": {"rows": None}})


def test_yahoo_chart_drops_the_unsettled_session_and_uses_exchange_dates() -> None:
    def stamp(day: str) -> int:
        return int(pd.Timestamp(day + " 13:30", tz="UTC").timestamp())  # 09:30 New York

    payload = {"chart": {"result": [{
        "meta": {"exchangeTimezoneName": "America/New_York"},
        "timestamp": [stamp("2026-09-30"), stamp("2026-10-01"), stamp("2026-10-02")],
        "indicators": {"quote": [{"close": [333.02, 330.32, 331.91]}]}}]}}
    during = us_prices.parse_yahoo_chart(payload, now=pd.Timestamp("2026-10-02 11:47", tz="America/New_York"))
    assert list(during.index) == [pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-01")]
    after = us_prices.parse_yahoo_chart(payload, now=pd.Timestamp("2026-10-02 18:00", tz="America/New_York"))
    assert len(after) == 3
    with pytest.raises(RuntimeError):
        us_prices.parse_yahoo_chart({"chart": {"result": None, "error": {"code": "Not Found"}}})


def test_us_close_falls_back_from_nasdaq_to_yahoo_and_skips_unknown_symbols(monkeypatch) -> None:
    from tsflab.realtime.sources import equity

    monkeypatch.setattr(equity.time, "sleep", lambda s: None)
    good = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-09-30", "2026-10-01"]))

    def nasdaq(symbol, start, end):
        raise RuntimeError("Nasdaq API returned no rows")

    def yahoo(symbol, start, end):
        if symbol == "NOPE":
            raise RuntimeError("no result")
        return good

    providers = [nasdaq, yahoo]
    start, end = pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-02")
    assert list(us_prices.close_with_fallback(providers, "AAPL", start, end)) == [1.0, 2.0]
    assert providers[0] is yahoo  # the working vendor moves first
    assert us_prices.close_with_fallback(providers, "NOPE", start, end).empty


def test_sp500_constituent_parsers() -> None:
    csv = "Symbol,Security\nMMM,3M\nBRK.B,Berkshire\nBF.B,Brown-Forman\n"
    assert sp500.parse_constituents_csv(csv) == ["BF-B", "BRK-B", "MMM"]
    wiki = "|| {{NyseSymbol|MMM}}\n|| [[3M]]\n|| {{NasdaqSymbol|AAPL}}\n|| {{NyseSymbol|BRK.B}}\n"
    assert sp500.parse_constituents_wikitext(wiki) == ["AAPL", "BRK-B", "MMM"]


def test_openaq_selection_spreads_countries_and_filters_reference_monitors(monkeypatch) -> None:
    def fake(path, params):
        if path == "/countries":
            return {"results": [{"code": "DE", "id": 1}, {"code": "FR", "id": 2}]}
        sensor = lambda i, name="pm25": {"id": i, "parameter": {"name": name}}  # noqa: E731
        results = {1: [{"isMonitor": True, "sensors": [sensor(11), sensor(12, "no2")]},
                       {"isMonitor": False, "sensors": [sensor(13)]},
                       {"isMonitor": True, "sensors": [sensor(14)]}],
                   2: [{"isMonitor": True, "sensors": [sensor(21)]}]}[params["countries_id"]]
        return {"results": results if params["page"] == 1 else []}

    monkeypatch.setattr(openaq, "_get", fake)
    assert openaq.select_sensors(["DE", "FR"], "pm25", 2, monitor_only=True) == ["11", "21"]
    assert openaq.select_sensors("DE", "pm25", 10, monitor_only=False) == ["11", "13", "14"]

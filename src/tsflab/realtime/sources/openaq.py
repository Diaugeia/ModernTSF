"""OpenAQ v3 hourly averages (free API key in ``OPENAQ_API_KEY``).

Channels are OpenAQ sensor ids for one parameter (default ``pm25``). The first
release selects up to ``max_sensors`` sensors in ``country`` / ``countries``
(optionally only reference monitors, ``monitor_only = true``); later releases
keep that channel set fixed.
"""

from __future__ import annotations

import time

import pandas as pd

from tsflab.realtime.sources import require_env
from tsflab.realtime.tracks import TrackSpec

API = "https://api.openaq.org/v3"


def _get(path: str, params: dict) -> dict:
    import requests

    key = require_env(["OPENAQ_API_KEY"])["OPENAQ_API_KEY"]
    for attempt in range(5):
        response = requests.get(f"{API}{path}", params=params, headers={"X-API-Key": key}, timeout=60)
        if response.status_code == 429:  # rate limited: back off
            time.sleep(10 * (attempt + 1))
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"OpenAQ rate limit persisted for {path}")


def select_sensors(countries: list[str] | str, parameter: str, limit: int, monitor_only: bool = False) -> list[str]:
    """Sensor ids measuring ``parameter`` in ``countries`` (ISO codes).

    ``monitor_only`` keeps reference-grade government monitors and drops
    low-cost sensors (OpenAQ's ``isMonitor`` location flag). Sensors are spread
    evenly over the countries so one dense network cannot fill the panel.
    """
    codes = [countries] if isinstance(countries, str) else list(countries)
    known = {c["code"]: c["id"] for c in _get("/countries", {"limit": 1000})["results"] if c.get("code")}
    per_country = max(1, -(-limit // len(codes)))
    sensors: list[str] = []
    for code in codes:
        found, page = [], 1
        while len(found) < per_country:
            batch = _get("/locations", {"countries_id": known[code], "limit": 1000, "page": page})["results"]
            if not batch:
                break
            for location in batch:
                if monitor_only and not location.get("isMonitor", False):
                    continue
                for sensor in location.get("sensors", []):
                    if sensor.get("parameter", {}).get("name") == parameter:
                        found.append(str(sensor["id"]))
            page += 1
        sensors.extend(sorted(found)[:per_country])
    return sorted(sensors)[:limit]


def parse_hours(results: list[dict]) -> pd.Series:
    """Turn a ``/sensors/{id}/hours`` result page into an hourly series (UTC, naive)."""
    index, values = [], []
    for row in results:
        stamp = row.get("period", {}).get("datetimeFrom", {}).get("utc")
        if stamp is None or row.get("value") is None:
            continue
        index.append(pd.Timestamp(stamp).tz_convert(None) if pd.Timestamp(stamp).tzinfo else pd.Timestamp(stamp))
        values.append(float(row["value"]))
    return pd.Series(values, index=pd.DatetimeIndex(index), dtype="float64")


def fetch(track: TrackSpec, start: pd.Timestamp, end: pd.Timestamp, channels: list[str] | None) -> pd.DataFrame:
    parameter = track.source.get("parameter", "pm25")
    sensors = channels or select_sensors(
        track.source.get("countries") or track.source.get("country", "CN"), parameter,
        int(track.source.get("max_sensors", 200)), bool(track.source.get("monitor_only", False)),
    )
    series = {}
    for sensor in sensors:
        rows, page = [], 1
        while True:
            chunk = _get(f"/sensors/{sensor}/hours", {
                "datetime_from": start.isoformat() + "Z", "datetime_to": end.isoformat() + "Z",
                "limit": 1000, "page": page,
            })["results"]
            rows.extend(chunk)
            if len(chunk) < 1000:
                break
            page += 1
        if rows:
            series[sensor] = parse_hours(rows)
    frame = pd.DataFrame(series).sort_index()
    return frame[~frame.index.duplicated()]

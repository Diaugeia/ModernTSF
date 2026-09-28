"""OpenAQ v3 hourly averages (free API key in ``OPENAQ_API_KEY``).

Channels are OpenAQ sensor ids for one parameter (default ``pm25``). The first
release selects up to ``max_sensors`` active sensors in the country; later
releases keep that channel set fixed.
"""

from __future__ import annotations

import time

import pandas as pd

from moderntsf.realtime.sources import require_env
from moderntsf.realtime.tracks import TrackSpec

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


def select_sensors(country: str, parameter: str, limit: int) -> list[str]:
    countries = _get("/countries", {"limit": 1000})["results"]
    country_id = next(c["id"] for c in countries if c.get("code") == country)
    sensors, page = [], 1
    while len(sensors) < limit:
        batch = _get("/locations", {"countries_id": country_id, "limit": 1000, "page": page})["results"]
        if not batch:
            break
        for location in batch:
            for sensor in location.get("sensors", []):
                if sensor.get("parameter", {}).get("name") == parameter:
                    sensors.append(str(sensor["id"]))
        page += 1
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
        track.source.get("country", "CN"), parameter, int(track.source.get("max_sensors", 200))
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

"""EIA-930 hourly grid data by balancing authority (free key in ``EIA_API_KEY``).

Register at https://www.eia.gov/opendata/ for a key. ``source.dataset`` picks the
series family:

- ``demand``: ``electricity/rto/region-data`` type ``D`` (hourly demand, MWh);
- ``solar``: ``electricity/rto/fuel-type-data`` fuel ``SUN`` (hourly utility-scale
  solar generation, MWh), only for balancing authorities that report it.

Hours are UTC. Reports are self-reported by the operators and revised later; the
panel keeps the first value it saw. EIA publishes US government data in the
public domain. Channels are individual balancing authorities (regional
aggregates are excluded): the best-covered ``max_series`` of the bootstrap window.
"""

from __future__ import annotations

import time

import pandas as pd

from moderntsf.realtime.sources import require_env
from moderntsf.realtime.tracks import TrackSpec

API = "https://api.eia.gov/v2/electricity/rto"
PAGE = 5000
# Regional and interconnection aggregates that overlap individual balancing authorities.
AGGREGATES = ["CAL", "CAR", "CENT", "FLA", "MIDA", "MIDW", "NE", "NW", "NY", "SE", "SW", "TEN", "TEX", "US48"]
_DATASETS = {
    "demand": ("region-data", {"facets[type][]": "D"}),
    "solar": ("fuel-type-data", {"facets[fueltype][]": "SUN"}),
}


def parse_rows(rows: list[dict]) -> pd.DataFrame:
    """EIA ``response.data`` rows -> wide hourly frame by respondent (UTC, naive)."""
    records = [(pd.to_datetime(r["period"], format="%Y-%m-%dT%H"), r["respondent"], float(r["value"]))
               for r in rows if r.get("value") not in (None, "")]
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records, columns=["time", "respondent", "value"])
    wide = frame.drop_duplicates(["time", "respondent"], keep="last").pivot(
        index="time", columns="respondent", values="value").sort_index()
    wide.index.name = None
    return wide


def _get(path: str, params: dict) -> dict:
    import requests

    params = {**params, "api_key": require_env(["EIA_API_KEY"])["EIA_API_KEY"]}
    for attempt in range(5):
        response = requests.get(f"{API}/{path}/data/", params=params, timeout=90)
        if response.status_code == 429:
            time.sleep(30 * (attempt + 1))
            continue
        response.raise_for_status()
        return response.json()["response"]
    raise RuntimeError("EIA rate limit persisted")


def fetch(track: TrackSpec, start: pd.Timestamp, end: pd.Timestamp, channels: list[str] | None) -> pd.DataFrame:
    path, facets = _DATASETS[track.source.get("dataset", "demand")]
    params: dict = {"frequency": "hourly", "data[0]": "value", "sort[0][column]": "period",
                    "sort[0][direction]": "asc", "length": PAGE,
                    "start": f"{start:%Y-%m-%dT%H}", "end": f"{end:%Y-%m-%dT%H}", **facets}
    if channels:
        params.update({f"facets[respondent][{i}]": name for i, name in enumerate(channels)})
    rows, offset = [], 0
    while True:
        chunk = _get(path, {**params, "offset": offset})["data"]
        rows.extend(chunk)
        if len(chunk) < PAGE:
            break
        offset += PAGE
        time.sleep(float(track.source.get("pause_seconds", 0.5)))
    panel = parse_rows(rows)
    if panel.empty or channels:
        return panel.reindex(columns=channels) if channels else panel
    coverage = panel.notna().mean().sort_values(ascending=False, kind="stable")
    coverage = coverage.drop(track.source.get("exclude", AGGREGATES), errors="ignore")
    keep = coverage[coverage >= float(track.source.get("min_coverage", 0.9))]
    return panel[sorted(keep.index[: int(track.source.get("max_series", 80))])]

"""UltraTraffic_CL archive: yearly hourly-flow panels per PeMS region (2003-2023).

Each ``<region>/<short>_Static/<year>.csv`` has a ``date`` column followed by
one column per station id. The bootstrap keeps the station set of the most
recent year, so the live PeMS increments extend exactly the same channels.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from moderntsf.realtime.tracks import TrackSpec


def _region_dir(track: TrackSpec) -> Path:
    env = track.bootstrap.get("root_env", "ULTRATRAFFIC_ROOT")
    root = os.environ.get(env)
    if not root:
        raise RuntimeError(f"set {env} to the extracted UltraTraffic_CL directory")
    region = track.bootstrap["region"]
    short = region.split("_", 1)[1]
    return Path(root) / region / f"{short}_{track.bootstrap.get('variant', 'Static')}"


def load(track: TrackSpec) -> pd.DataFrame:
    directory = _region_dir(track)
    years = sorted(int(p.stem) for p in directory.glob("[0-9][0-9][0-9][0-9].csv"))
    first = int(track.bootstrap.get("first_year", years[0]))
    years = [y for y in years if y >= first]
    if not years:
        raise FileNotFoundError(f"no yearly panels under {directory}")
    latest = pd.read_csv(directory / f"{years[-1]}.csv", nrows=0).columns[1:]
    frames = []
    for year in years:
        frame = pd.read_csv(directory / f"{year}.csv", index_col="date", parse_dates=True)
        frame.columns = [str(c) for c in frame.columns]
        frames.append(frame.reindex(columns=[str(c) for c in latest]))
    return pd.concat(frames).sort_index()

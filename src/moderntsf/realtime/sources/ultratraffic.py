"""Bootstrap a traffic track from the local UltraTraffic parquet store.

The store (``dataset/ultratraffic``, built by
:mod:`moderntsf.data.prepare.ultratraffic`) is the same one the static
``ultratraffic_*`` datasets read, so a track's history and the static presets
share one loader. The bootstrap keeps the station set of the most recent year,
so live PeMS increments extend exactly the same channels.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from moderntsf.data.ultratraffic_store import load_panel
from moderntsf.realtime.tracks import TrackSpec


def load(track: TrackSpec) -> pd.DataFrame:
    root = Path(os.environ.get(track.bootstrap.get("root_env", "ULTRATRAFFIC_ROOT"), "dataset/ultratraffic"))
    region = track.bootstrap["region"]
    available = sorted(int(p.stem) for p in (root / region / "static").glob("[0-9][0-9][0-9][0-9].parquet"))
    if not available:
        raise FileNotFoundError(f"no UltraTraffic store for {region} under {root}; run "
                                "`python -m moderntsf.data.prepare.ultratraffic --archive TrafficCL.zip`")
    first = int(track.bootstrap.get("first_year", available[0]))
    years = [y for y in available if y >= first]
    return load_panel(str(root), region, years, "static", "last")

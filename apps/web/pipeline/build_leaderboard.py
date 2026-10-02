#!/usr/bin/env python3
"""Build data/leaderboard.json from submissions/ (+ a curated overlay).

Validation and aggregation are the shared TSF-Core implementation
(``tsflab.tsf_core.leaderboard``); this script only adds the site's
presentation layer:

  1. map canonical (track, dataset_id) keys to the site's display keys;
  2. overlay curated blocks that submissions do not cover yet (air quality and
     the stock *quant* view);
  3. attach the rolling real-time summaries from data/realtime/*.json.

Usage:
  python pipeline/build_leaderboard.py             # validate + aggregate + write
  python pipeline/build_leaderboard.py --no-write  # dry run: summary only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent.parent / "src"))

from tsflab.tsf_core.leaderboard import PRIMARY_METRIC, aggregate, load_submissions  # noqa: E402

BOARD = ROOT / "data" / "leaderboard.json"
REALTIME = ROOT / "data" / "realtime"

# (canonical track, dataset_id) -> (site track key, display name)
DISPLAY = {("realtime", "stock_hs300"): ("stock", "Stock-HS300")}


def to_display(tracks: dict) -> dict:
    out: dict = {}
    for track, block in tracks.items():
        for dataset, data in block["datasets"].items():
            site_track, name = DISPLAY.get((track, dataset), (track, dataset))
            out.setdefault(site_track, {"datasets": {}})["datasets"][name] = data
    return out


def overlay_curated(tracks: dict, curated: dict) -> dict:
    out = json.loads(json.dumps(tracks))
    for track, block in curated.get("tracks", {}).items():
        for dataset, data in block.get("datasets", {}).items():
            target = out.setdefault(track, {"datasets": {}})["datasets"]
            if dataset not in target:
                target[dataset] = data  # no submissions produced this block yet
            elif "quant" in data:
                target[dataset]["quant"] = data["quant"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true", help="dry run: summary only")
    args = parser.parse_args()

    valid, rejected = load_submissions(ROOT / "submissions")
    if rejected:
        print(f"❌ {len(rejected)} invalid submission(s) — run pipeline/validate.py; board not updated.")
        return 1
    curated = json.loads(BOARD.read_text(encoding="utf-8")) if BOARD.is_file() else {}
    tracks = overlay_curated(to_display(aggregate(doc for _, doc in valid)), curated)
    realtime = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(REALTIME.glob("*.json"))}
    board = {
        "schema_version": "1.1",
        "generated_at": curated.get("generated_at"),
        "primary_metric": PRIMARY_METRIC,
        "n_submissions": len(valid),
        "n_rejected": 0,
        "tracks": tracks,
        "realtime": realtime,
    }
    print(f"Aggregated {len(valid)} submission files:")
    for track, block in tracks.items():
        for dataset, data in block["datasets"].items():
            for horizon, rows in data["horizons"].items():
                multi = sum(1 for r in rows if r.get("n_runs", 1) > 1)
                print(f"  {track}/{dataset}/h={horizon}: {len(rows)} models" + (f", {multi} multi-run" if multi else ""))
    for track, summary in realtime.items():
        print(f"  realtime/{track}: {len(summary.get('scored_rounds', []))} scored round(s)")
    if args.no_write:
        print("\n(dry run — not written)")
        return 0
    BOARD.write_text(json.dumps(board, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"\n✅ wrote {BOARD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

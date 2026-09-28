#!/usr/bin/env python3
"""Build data/leaderboard.json by aggregating submissions/ (+ a curated overlay).

Pipeline:
  1. validate every submission against the TSF-Core contract (pipeline/validate.py);
  2. aggregate submissions/**/submission.json into leaderboard rows, averaging
     metrics across runs (seeds) per (track, dataset, horizon, model) and recording
     n_runs / std / submission_ids;
  3. overlay curated rows for blocks that have no raw submissions yet (air-quality,
     and the stock *quant* view), so those tracks don't regress.

Two submission shapes are accepted (same record fields in both):
  - flat:   {model, dataset_id, track, seed, results:[{horizon, metrics, timing}]}
  - bundle: {manifest, datasets, records:[ <flat record>, ... ]}

Usage:
  python pipeline/build_leaderboard.py            # validate + aggregate + write
  python pipeline/build_leaderboard.py --no-write # dry run: print a summary only
  python pipeline/build_leaderboard.py --no-validate
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBS = os.path.join(ROOT, "submissions")
BOARD = os.path.join(ROOT, "data", "leaderboard.json")

PRIMARY_METRIC = "mse"
# Metric columns surfaced on the board (lower-is-better except corr).
METRIC_KEYS = ["mse", "mae", "rmse", "corr", "wape", "rse"]

# submission dataset_id -> leaderboard display name (defaults to itself).
DATASET_DISPLAY = {"stock_hs300": "Stock-HS300"}


def leaderboard_track(track: str, dataset_id: str) -> str:
    """Map a submission's (track, dataset_id) to a leaderboard track key."""
    if dataset_id == "stock_hs300":
        return "stock"
    return track  # time_series → time_series


def iter_records(doc: dict):
    """Yield flat records from either submission shape."""
    if "records" in doc and isinstance(doc["records"], list):
        yield from doc["records"]
    elif "model" in doc and "results" in doc:
        yield doc


def submission_id(doc: dict, rec: dict) -> str:
    return (
        rec.get("record_id")
        or (doc.get("manifest") or {}).get("submission_id")
        or rec.get("model", "?")
    )


def mean_std(values: list):
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return round(m, 6), None
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return round(m, 6), round(math.sqrt(var), 6)


def aggregate() -> dict:
    # (lb_track, dataset, horizon, model) -> list of per-run metric dicts (+ _sid)
    groups: dict = defaultdict(list)
    n_files = 0
    for f in glob.glob(os.path.join(SUBS, "**", "submission.json"), recursive=True):
        try:
            doc = json.load(open(f, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  ! skip unreadable {f}: {e}", file=sys.stderr)
            continue
        n_files += 1
        for rec in iter_records(doc):
            model = rec.get("model")
            ds_id = rec.get("dataset_id")
            if not model or not ds_id:
                continue
            lb_track = leaderboard_track(rec.get("track", ""), ds_id)
            dataset = DATASET_DISPLAY.get(ds_id, ds_id)
            sid = submission_id(doc, rec)
            for r in rec.get("results", []):
                h = str(r.get("horizon"))
                metrics = r.get("metrics") or {}
                timing = r.get("timing") or {}
                groups[(lb_track, dataset, h, model)].append(
                    {
                        **{k: metrics.get(k) for k in METRIC_KEYS},
                        "fit_time": timing.get("fit_time_sec"),
                        "inference_time": timing.get("inference_time_sec"),
                        "_sid": sid,
                    }
                )

    # Collapse each group into one aggregated row.
    by_block: dict = defaultdict(list)  # (track, dataset) -> [(horizon, row)]
    for (lb_track, dataset, h, model), runs in groups.items():
        row: dict = {"model": model}
        for k in METRIC_KEYS:
            m, s = mean_std([r.get(k) for r in runs])
            row[k] = m
            if s is not None:
                row[f"{k}_std"] = s
        row["n_runs"] = len(runs)
        row["submission_ids"] = sorted({r["_sid"] for r in runs})
        by_block[(lb_track, dataset)].append((h, row))

    tracks: dict = {}
    for (lb_track, dataset), entries in by_block.items():
        horizons: dict = defaultdict(list)
        for h, row in entries:
            horizons[h].append(row)
        # Rank each horizon by the primary metric (lower is better).
        for rows in horizons.values():
            rows.sort(key=lambda r: (r.get(PRIMARY_METRIC) is None, r.get(PRIMARY_METRIC)))
            for i, r in enumerate(rows, 1):
                r["rank"] = i
        ordered = {h: horizons[h] for h in sorted(horizons, key=lambda x: int(x) if x.isdigit() else x)}
        tracks.setdefault(lb_track, {"datasets": {}})["datasets"][dataset] = {"horizons": ordered}
    return {"tracks": tracks, "n_files": n_files}


def overlay_curated(agg_tracks: dict) -> dict:
    """Keep curated blocks (and the stock quant view) that submissions don't cover."""
    curated = json.load(open(BOARD, encoding="utf-8"))
    out = json.loads(json.dumps(agg_tracks))  # deep copy

    for track, tblock in curated.get("tracks", {}).items():
        for dataset, dblock in tblock.get("datasets", {}).items():
            dst = out.setdefault(track, {"datasets": {}})["datasets"]
            if dataset not in dst:
                # No submissions produced this block → keep curated as-is.
                dst[dataset] = dblock
            elif "quant" in dblock:
                # Block was aggregated from submissions; carry over the curated
                # quant view (no raw quant submissions exist yet).
                dst[dataset]["quant"] = dblock["quant"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--no-write", action="store_true", help="dry run: summary only")
    args = ap.parse_args()

    if not args.no_validate:
        rc = subprocess.call([sys.executable, os.path.join(ROOT, "pipeline", "validate.py")])
        if rc != 0:
            print("❌ submission validation failed — not updating the board.")
            return rc

    agg = aggregate()
    tracks = overlay_curated(agg["tracks"])

    board = {
        "schema_version": "1.0",
        "generated_at": json.load(open(BOARD, encoding="utf-8")).get("generated_at"),
        "primary_metric": PRIMARY_METRIC,
        "n_submissions": agg["n_files"],
        "n_rejected": 0,
        "tracks": tracks,
    }

    print(f"\nAggregated {agg['n_files']} submission files:")
    for track, tb in tracks.items():
        for ds, db in tb["datasets"].items():
            for h, rows in db["horizons"].items():
                multi = sum(1 for r in rows if r.get("n_runs", 1) > 1)
                tag = f", {multi} multi-run" if multi else ""
                print(f"  {track}/{ds}/h={h}: {len(rows)} models{tag}")

    if args.no_write:
        print("\n(dry run — not written)")
        return 0

    with open(BOARD, "w", encoding="utf-8") as f:
        json.dump(board, f, ensure_ascii=False, separators=(",", ":"))
    print(f"\n✅ wrote {os.path.relpath(BOARD, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

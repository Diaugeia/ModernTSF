#!/usr/bin/env python3
"""tsf leaderboard-build — aggregate submissions into a leaderboard.json.

Reads every ``submission.json`` under a submissions root (local dir or a synced
HF dataset), checks each is present + schema-valid, then collates the scattered
results into one ranked ``leaderboard.json`` keyed by track / dataset / horizon.

Deterministic, no LLM, no torch — depends only on ``tsf_core`` (for the schema)
+ stdlib, with ``jsonschema`` used when available.

    uv run tsf leaderboard-build --source work_dirs/_submissions
    uv run tsf leaderboard-build --source <dir> --out leaderboard.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "1.0"
PRIMARY_METRIC = "mse"  # lower is better; ranking key
# Metrics carried per leaderboard row (each is present in every RunRecord's
# MetricSet). `mse` stays first/required; the rest are averaged when present.
METRIC_FIELDS = ("mse", "mae", "rmse", "corr")

# Tracks that always appear in the leaderboard, in this order — even with zero
# submissions — so the two-tier structure (static + realtime) stays visible and
# empty tracks read as "open for submissions" rather than silently disappearing.
CANONICAL_TRACKS = ("time_series", "spatiotemporal", "covariate", "realtime")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _schema_path() -> Path | None:
    try:
        from moderntsf import tsf_core
        return Path(tsf_core.__file__).parent / "schema" / "submission_report.schema.json"
    except Exception:
        return None


def _validator():
    """Return a jsonschema validator, or None if unavailable (skip validation)."""
    sp = _schema_path()
    if sp is None or not sp.exists():
        return None
    try:
        from jsonschema import Draft202012Validator

        return Draft202012Validator(json.loads(sp.read_text()))
    except Exception:
        return None


def _load_submissions(source: Path, validator) -> tuple[list[dict], list[str]]:
    """Return (valid submission dicts, rejection messages)."""
    subs: list[dict] = []
    rejects: list[str] = []
    # Recursive: supports both flat `<id>/submission.json` and nested
    # `<track>/<dataset>/<model>/<id>/submission.json` layouts.
    for sj in sorted(source.rglob("submission.json")):
        sid = sj.parent.name
        try:
            data = json.loads(sj.read_text())
        except Exception as exc:
            rejects.append(f"{sid}: unreadable submission.json ({exc})")
            continue
        # Minimal v1 check: a result + a trajectory must both be present.
        if not data.get("records"):
            rejects.append(f"{sid}: no records (result missing)")
            continue
        if not data.get("trajectories"):
            rejects.append(f"{sid}: no trajectory")
            continue
        if validator is not None:
            errs = sorted(validator.iter_errors(data), key=lambda e: e.path)
            if errs:
                rejects.append(f"{sid}: schema invalid — {errs[0].message}")
                continue
        subs.append(data)
    return subs, rejects


def _collate(subs: list[dict]) -> dict:
    """Group results into cells keyed by (track, dataset, model, horizon)."""
    cells: dict[tuple, dict] = {}
    for sub in subs:
        sid = sub.get("manifest", {}).get("submission_id", "?")
        for rec in sub.get("records", []):
            track = rec.get("track")
            dataset = rec.get("dataset_id")
            model = rec.get("model")
            for hr in rec.get("results", []):
                horizon = hr.get("horizon")
                m = hr.get("metrics", {})
                key = (track, dataset, model, horizon)
                cell = cells.setdefault(
                    key, {name: [] for name in METRIC_FIELDS} | {"submission_ids": set()}
                )
                for name in METRIC_FIELDS:
                    v = m.get(name)
                    if isinstance(v, (int, float)):
                        cell[name].append(float(v))
                cell["submission_ids"].add(sid)
    return cells


def _build_leaderboard(cells: dict, primary: str) -> dict:
    """Average across seeds/submissions, then rank models per (track,dataset,horizon)."""
    tracks: dict[str, dict] = {}
    # First fold each cell to one averaged entry.
    folded: dict[tuple, dict] = {}
    for (track, dataset, model, horizon), agg in cells.items():
        if not agg["mse"]:
            continue
        entry = {
            "model": model,
            "n_runs": len(agg["mse"]),
            "submission_ids": sorted(agg["submission_ids"]),
        }
        for name in METRIC_FIELDS:
            entry[name] = round(statistics.fmean(agg[name]), 6) if agg[name] else None
        folded[(track, dataset, model, horizon)] = entry
    # Then rank models within each (track, dataset, horizon).
    groups: dict[tuple, list] = {}
    for (track, dataset, model, horizon), entry in folded.items():
        groups.setdefault((track, dataset, horizon), []).append(entry)
    for (track, dataset, horizon), entries in groups.items():
        entries.sort(key=lambda e: e[primary])
        for i, e in enumerate(entries, 1):
            e["rank"] = i
        (tracks.setdefault(track, {"datasets": {}})["datasets"]
         .setdefault(dataset, {"horizons": {}})["horizons"][str(horizon)]) = entries
    return tracks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tsf leaderboard-build", description=__doc__)
    ap.add_argument("--source", default="work_dirs/_submissions",
                    help="Directory of submission bundles (each <id>/submission.json)")
    ap.add_argument("--out", default="leaderboard.json", help="Output path")
    ap.add_argument("--primary-metric", default=PRIMARY_METRIC,
                    help="Metric used to rank (lower is better; default: mse)")
    args = ap.parse_args(argv)
    primary = args.primary_metric

    source = Path(args.source)
    if not source.is_dir():
        sys.exit(f"error: source dir not found: {source}")

    validator = _validator()
    subs, rejects = _load_submissions(source, validator)
    cells = _collate(subs)
    tracks = _build_leaderboard(cells, primary)

    # Keep the two-tier structure visible: every canonical track is present
    # (empty when it has no submissions yet), in a stable order; any extra track
    # seen in the data is appended after.
    ordered = {t: tracks.get(t, {"datasets": {}}) for t in CANONICAL_TRACKS}
    for t, block in tracks.items():
        if t not in ordered:
            ordered[t] = block
    tracks = ordered

    leaderboard = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "primary_metric": primary,
        "n_submissions": len(subs),
        "n_rejected": len(rejects),
        "rejections": rejects,
        "tracks": tracks,
    }
    Path(args.out).write_text(json.dumps(leaderboard, indent=2, ensure_ascii=False))

    n_cells = sum(len(h) for t in tracks.values() for d in t["datasets"].values()
                  for h in d["horizons"].values())
    print(f"Built {args.out}")
    print(f"  submissions: {len(subs)} accepted, {len(rejects)} rejected"
          + (" (no jsonschema validation — install jsonschema)" if validator is None else ""))
    print(f"  tracks: {', '.join(tracks) or '(none)'} · ranked entries: {n_cells}")
    for msg in rejects:
        print(f"  reject: {msg}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

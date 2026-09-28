#!/usr/bin/env python3
"""tsf submit — package a run into a self-contained Submission Report.

Locates a run's ``record.json`` (+ ``profile.csv`` + research events) and
assembles a schema-valid ``moderntsf.tsf_core.SubmissionReport`` bundle (machine results +
audit trajectory + human report) under ``work_dirs/_submissions/``. Depends only
on ``tsf_core`` + stdlib.

The ModernTSF Leaderboard is GitHub-canonical: contribute the bundle via a pull
request on https://github.com/Diaugeia/ModernTSF (under ``apps/web/submissions/``). There is
no Hugging Face Submissions dataset — ``tsf submit`` builds the evidence; you
open the PR.

Examples
--------
    uv run tsf submit --dataset ETTh1 --model iTransformer --latest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from moderntsf.tsf_core.paths import working_root

ROOT = working_root()
WORK = ROOT / "work_dirs"

# Submissions are contributed to this GitHub repo via PR (GitHub-canonical).
LEADERBOARD_REPO = "https://github.com/Diaugeia/ModernTSF"
PROFILE_INT_FIELDS = {"total_params", "trainable_params", "non_trainable_params"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_\-]+", "_", (text or "").lower()).strip("_") or "x"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_user() -> str | None:
    try:
        r = subprocess.run(
            ["git", "config", "user.name"], cwd=ROOT, capture_output=True, text=True, timeout=5
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def _find_record(dataset: str, model: str, run_id: str | None) -> Path:
    rdir = WORK / dataset / model / "records"
    if not rdir.exists():
        sys.exit(f"error: no records dir at {rdir} — run an experiment first (records appear there).")
    if run_id:
        p = rdir / f"{run_id}.json"
        if not p.exists():
            sys.exit(f"error: record not found: {p}")
        return p
    files = sorted(rdir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit(f"error: no record.json under {rdir}")
    return files[-1]


def _load_profile(dataset: str, model: str, run_id: str) -> dict | None:
    pf = WORK / dataset / model / "profile.csv"
    if not pf.exists() or not run_id:
        return None
    try:
        with open(pf, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("run_id") != run_id:
                    continue
                out: dict = {}
                from moderntsf.tsf_core import PROFILE_FIELDS

                for k in PROFILE_FIELDS:
                    v = row.get(k)
                    if v in (None, ""):
                        continue
                    if k in PROFILE_INT_FIELDS:
                        try:
                            out[k] = int(float(v))
                        except ValueError:
                            continue
                    else:
                        out[k] = v
                return out or None
    except Exception:
        return None
    return None


def _gather_trajectory(dest: Path, run_id: str) -> dict:
    """Collect research events for ``run_id`` or synthesize a minimal record."""
    from moderntsf.benchmark.research_round import events_for_run

    events = events_for_run(run_id)
    synthetic = not events
    if synthetic:
        events = [
            {
                "synthetic": True,
                "note": "No research round was associated with this experiment. "
                "Reconstructed from run artifacts.",
                "run_id": run_id,
                "time": _now(),
            }
        ]
    events.sort(key=lambda event: event.get("time", ""))
    with open(dest, "w") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
    return {"n_events": len(events), "synthetic": synthetic}


def _render_report(record, ds_spec) -> str:
    """Human-readable Markdown report for one submission (renders natively on the
    HF dataset viewer and stays diff-friendly inside the Submissions repo)."""
    def cell(v):
        return "—" if v is None else str(v)

    rows = "\n".join(
        f"| {hr.horizon} | {cell(hr.metrics.mse)} | {cell(hr.metrics.mae)} | "
        f"{cell(hr.metrics.rmse)} | {cell(hr.metrics.corr)} | `{hr.run_id}` |"
        for hr in record.results
    )
    prof = record.results[0].profile
    prof_lines = "(none)"
    if prof is not None:
        items = {k: v for k, v in prof.model_dump().items() if v is not None}
        if items:
            prof_lines = " · ".join(f"{k} {v}" for k, v in items.items())
    env = record.env.model_dump()
    env_line = " · ".join(f"{k} {v}" for k, v in env.items() if v is not None)
    return f"""# ModernTSF Leaderboard Submission — {record.model} / {record.dataset_id}

- **track** `{record.track}` · **mode** `{record.mode}` · **seed** {record.seed}
- **dataset** `{ds_spec.id}@{ds_spec.version}`
- **created** {record.created_at or ""}

## Metrics by horizon

| pred_len | MSE | MAE | RMSE | Corr | run_id |
|---:|---|---|---|---|---|
{rows}

## Profile

{prof_lines}

## Environment

{env_line}

_Generated by `tsf submit` · Diaugeia / ModernTSF Leaderboard._
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tsf submit", description=__doc__)
    ap.add_argument("--dataset", required=True, help="Dataset name (work_dirs/<dataset>/...)")
    ap.add_argument("--model", required=True, help="Model name (work_dirs/<dataset>/<model>/...)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--run-id", help="Specific run_id to submit")
    g.add_argument("--latest", action="store_true", help="Use the newest record.json (default)")
    ap.add_argument("--track", default=None, help="Override track (default: from record)")
    ap.add_argument("--submitter", default=None, help="Submitter name (default: git user)")
    ap.add_argument("--dataset-version", default="1.0.0", help="DatasetSpec version to pin")
    ap.add_argument("--out-dir", default=None, help="Output dir (default: work_dirs/_submissions)")
    args = ap.parse_args(argv)

    from moderntsf import tsf_core
    rec_path = _find_record(args.dataset, args.model, args.run_id)
    rec_data = json.loads(rec_path.read_text())
    if not rec_data.get("results"):
        sys.exit(f"error: record has no results: {rec_path}")
    run_id = rec_data["results"][0].get("run_id") or rec_path.stem

    # Enrich the first horizon with profiling, if available.
    prof = _load_profile(args.dataset, args.model, run_id)
    if prof:
        rec_data["results"][0]["profile"] = prof

    # Validate the (enriched) record against the contract.
    try:
        record = tsf_core.RunRecord(**rec_data)
    except Exception as exc:
        sys.exit(f"error: record.json failed contract validation: {exc}")

    track = args.track or record.track
    submitter = args.submitter or _git_user() or "anonymous"
    submission_id = f"{_slug(submitter)}__{run_id}"

    ds_spec = tsf_core.DatasetSpec(
        id=_slug(args.dataset),
        version=args.dataset_version,
        mode=record.mode,
        track=track,
    )

    out_root = Path(args.out_dir) if args.out_dir else WORK / "_submissions"
    sub_dir = out_root / submission_id
    sub_dir.mkdir(parents=True, exist_ok=True)

    # 1) trajectory.jsonl  2) report.md  -> hash both
    traj_path = sub_dir / "trajectory.jsonl"
    traj_info = _gather_trajectory(traj_path, run_id)
    traj_sha = _sha256_file(traj_path)

    report_path = sub_dir / "report.md"
    report_md = _render_report(record, ds_spec)
    report_path.write_text(report_md)
    report_sha = _sha256_file(report_path)

    files = [
        tsf_core.FileRef(path="trajectory.jsonl", sha256=traj_sha, bytes=traj_path.stat().st_size, role="trajectory"),
        tsf_core.FileRef(path="report.md", sha256=report_sha, bytes=report_path.stat().st_size, role="report"),
    ]
    files_sha = _sha256_bytes(
        json.dumps([f.model_dump() for f in files], sort_keys=True).encode()
    )
    manifest = tsf_core.SubmissionManifest(
        submission_id=submission_id,
        submitter=submitter,
        track=track,
        created_at=_now(),
        files=files,
        files_sha256=files_sha,
    )
    report = tsf_core.SubmissionReport(
        manifest=manifest,
        datasets=[ds_spec],
        records=[record],
        trajectories=[
            tsf_core.TrajectoryRef(
                path="trajectory.jsonl", sha256=traj_sha,
                synthetic=traj_info["synthetic"], n_events=traj_info["n_events"],
            )
        ],
        reports=[tsf_core.ReportArtifact(path="report.md", sha256=report_sha, format="md")],
    )

    (sub_dir / "submission.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False)
    )

    print(f"Submission Report built: {sub_dir}")
    print(f"  submission_id : {submission_id}")
    print(f"  track/dataset : {track} / {ds_spec.id}@{ds_spec.version}")
    print(f"  records       : 1 ({len(record.results)} horizon[s])")
    print(f"  trajectory    : {traj_info['n_events']} event(s)"
          + (" [SYNTHETIC — use `tsf run --round <id>` next time]" if traj_info["synthetic"] else ""))
    print("  files         : submission.json, trajectory.jsonl, report.md")

    # The leaderboard is GitHub-canonical: contribute the bundle via a PR on the
    # ModernTSF repo, under the nested append-only layout that `leaderboard-build`
    # scans recursively. There is no Hugging Face Submissions dataset.
    dest = f"apps/web/submissions/{track}/{_slug(ds_spec.id)}/{_slug(record.model)}/{submission_id}"

    print("\nNext — add this bundle to the ModernTSF Leaderboard via a GitHub PR:")
    print(f"  1. clone {LEADERBOARD_REPO}")
    print(f"  2. copy the bundle dir into {dest}/")
    print("  3. open a PR — CI validates, aggregates, and redeploys the board")
    print("  (exact steps + accepted formats: apps/web/SUBMITTING.md in that repo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

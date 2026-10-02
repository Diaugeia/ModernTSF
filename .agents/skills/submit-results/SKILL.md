---
name: submit-results
description: "Package a completed TSFLab run and its research evidence as a TSFLab Leaderboard submission bundle. Use for local submission bundles or leaderboard contribution; opening a pull request requires explicit authorization. Not for live real-time forecasts (forecast-realtime-round) or weights (publish-weights)."
---

# Submit results

Release module: turn a finished run into a validated submission bundle that the leaderboard can
recompute from.

## Inputs

- A completed run (`work_dirs/<dataset>/<model>/records/<run_id>.json`), ideally
  executed inside a research round so its trajectory is captured. Runs execute on a
  GPU machine or CI (see `run-experiment`); packaging happens locally from the
  returned `work_dirs/` and does no training.

## Steps

1. Open a round before the run when possible (`tsf research start --task submission
   --goal <goal> --max-runs <count>`, then `tsf run <cfg> --round <id>` on the run
   machine, then `tsf research status <id> completed --message <conclusion>`). Without
   one the bundle's trajectory is marked synthetic. Package the finished record:

   ```bash
   uv run tsf result submit --dataset <dataset> --model <model> [--run-id <run_id> | --latest]
   ```
   The bundle lands in `work_dirs/_submissions/`; `--latest` takes the newest record.

2. Inspect `submission.json`, `trajectory.jsonl`, and `report.md`. Confirm dataset
   version, run identity, metrics, and whether the trajectory is synthetic.
3. Place the bundle under `apps/web/submissions/<track>/<dataset>/<model>/<id>/`
   and check it with the same contract the site uses:

   ```bash
   uv run tsf result leaderboard --source apps/web/submissions --out work_dirs/_board/leaderboard.json
   uv run tsf repo check --only web-submissions schema-export
   ```

4. Weights are never part of a submission; publish them separately with
   `publish-weights` when reproducibility needs them.

## Chain

- Module: Release.
- Reads: a completed record, its round trajectory, the result board.
- Produces: a bundle under `apps/web/submissions/<track>/<dataset>/<model>/<id>/`; its rows rebuild the leaderboard.
- Hands off to: `run-autoresearch` reads the leaderboard as a reference bar; weights go to `publish-weights`.

## Success

- A contract-valid bundle whose rows appear in the locally rebuilt board.

## Stop and hand off

- A branch, issue, or pull request requires explicit approval.
- Forecasts for live rounds use `forecast-realtime-round`, not this skill.

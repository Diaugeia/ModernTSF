---
name: submit-results
description: Package a completed TSFLab run and its research evidence as a TSFLab Leaderboard submission. Use for local submission bundles or leaderboard contribution; publishing a pull request requires explicit authorization.
---

# Submit results

Release module: turn a finished run into a validated submission bundle that the leaderboard can
recompute from.

## Inputs

- A completed run (`work_dirs/<dataset>/<model>/records/<run_id>.json`), ideally
  executed inside a research round so its trajectory is captured.

## Steps

1. Run inside a round when possible, then package:

   ```bash
   uv run tsf research start --task submission --goal <goal> --max-runs <count>
   uv run tsf run configs/runs/<run>.toml --round <round-id>
   uv run tsf research status <round-id> completed --message <conclusion>
   uv run tsf result submit --dataset <dataset> --model <model> --latest
   ```

2. Inspect `submission.json`, `trajectory.jsonl`, and `report.md`. Confirm dataset
   version, run identity, metrics, and whether the trajectory is synthetic.
3. Place the bundle under `apps/web/submissions/<track>/<dataset>/<model>/<id>/`
   and check it with the same contract the site uses:

   ```bash
   uv run tsf result leaderboard --source apps/web/submissions --out /tmp/board.json
   uv run tsf repo schema --check
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

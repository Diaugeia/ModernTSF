---
name: submit-results
description: Package a completed ModernTSF run and its research evidence for ModernTSF Leaderboard submission. Use for local submission bundles or leaderboard contribution; publishing a pull request requires explicit authorization.
---

# Submit results

```bash
uv run tsf research start --task submission --goal <goal> --max-runs <count>
uv run tsf run configs/runs/<run>.toml --round <round-id>
uv run tsf research status <round-id> completed --message <conclusion>
uv run tsf submit --dataset <dataset> --model <model> --latest
uv run tsf schema-export --check
```

Inspect `submission.json`, `trajectory.jsonl`, and `report.md`; the trajectory file
is the ModernTSF Leaderboard export of matching research events, not a second local state system.
Confirm dataset version, run identity, metrics, and synthetic status. Use
`uv run tsf leaderboard-build` for a local board. A remote branch, issue, or pull
request requires explicit user approval.

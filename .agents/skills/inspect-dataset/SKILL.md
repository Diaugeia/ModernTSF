---
name: inspect-dataset
description: Inspect, profile, or visualize an existing TSFLab dataset. Use for resolved shapes, trend and seasonality characteristics, split checks, leakage checks, model-selection profiles, or raw sample plots; not for result plots or model predictions.
---

# Inspect a dataset

Data module, step two: report what a dataset preset actually loads, separating
observations from inferred characteristics. The `analyze` profile is the data
context AutoResearch reads before it picks models.

## Inputs

- A dataset preset (`configs/datasets/<name>.toml`) and the split of interest.

## Steps

```bash
uv run tsf dataset show <name>
uv run tsf dataset inspect --config configs/datasets/<name>.toml --split train --per-channel
uv run tsf dataset analyze <name>        # structured profile for model selection
uv run tsf dataset plot --config configs/datasets/<name>.toml --split train --num-samples 3
```

Start from the card's front matter (`domain`, `frequency`, `length`, `channels`,
`protocol`) and its "Standard protocol and known pitfalls", then confirm them
against the files. `analyze` writes `work_dirs/profiles/<name>/profile.{json,md}` from train statistics plus
labelled train/val/test shift; use it before choosing models.

Check split boundaries, tensor dimensions, missing values, target-channel
behavior, leakage across splits, inferred seasonal period, and adjacency or
covariate metadata.

## Chain

- Module: Data.
- Reads: dataset card (L1, then L2) and the loaded splits.
- Produces: profile `work_dirs/profiles/<name>/profile.{json,md}` and a measured-fact report.
- Hands off to: `run-autoresearch` (profile-driven model choice), `run-experiment` (design), `add-dataset` (card fixes).

## Success

- A short report with artifact paths, the facts observed, and which
  characteristics are inferred.
- Any card fact the data contradicts is reported with the measured value; fix the
  card (`stats_basis: "measured"`) only when asked to update it.

## Stop and hand off

- Inspection never modifies data. Fixes belong to `add-dataset`; model output
  plots belong to `analyze-results`; profile-driven search to `run-autoresearch`.

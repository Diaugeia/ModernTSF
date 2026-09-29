---
name: inspect-dataset
description: Inspect, profile, or visualize an existing ModernTSF dataset. Use for resolved shapes, trend and seasonality characteristics, split checks, leakage checks, or raw sample plots; not for result plots or model predictions.
---

# Inspect a dataset

Report what a dataset preset actually loads, separating observations from
inferred characteristics.

## Inputs

- A dataset preset (`configs/datasets/<name>.toml`) and the split of interest.

## Steps

```bash
uv run tsf dataset show <name>
uv run tsf dataset inspect --config configs/datasets/<name>.toml --split train --per-channel
uv run tsf dataset plot --config configs/datasets/<name>.toml --split train --num-samples 3
```

Check split boundaries, tensor dimensions, missing values, target-channel
behavior, leakage across splits, inferred seasonal period, and adjacency or
covariate metadata.

## Success

- A short report with artifact paths, the facts observed, and which
  characteristics are inferred.

## Stop and hand off

- Inspection never modifies data. Fixes belong to `prepare-dataset` or
  `add-dataset`; model output plots belong to `analyze-results`.

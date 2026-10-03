---
name: "rt/grid_eia_us"
description: "Hourly electricity demand of up to 80 US balancing authorities from EIA-930, frozen release of real-time track grid_eia_us; seq_len 168, horizon 24. Use for multivariate or spatiotemporal hourly load forecasting across regions; not for multi-year seasonality (365-day bootstrap) or overlapping regional aggregates."
---

# rt/grid_eia_us

## Overview

`rt_grid_eia_us` serves the EIA-930 demand real-time track as a static spatiotemporal dataset: hourly demand of up to 80 US balancing authorities (those reporting at least 90% of hours at bootstrap) as nodes, from the EIA Hourly Electric Grid Monitor. History is a 365-day API backfill plus each weekly release. Use it to train and check models before submitting to the live track.

## Protocol and pitfalls

- **Operator-reported.** A few outliers and gaps are normal; regional aggregates such as `US48` or `TEX` are excluded so channels do not overlap.
- **Reporting lag.** Demand lags about an hour in the live feed.
- **Frozen snapshot, not the live track.** No `version` is pinned, so the preset reads whatever the local panel store holds; pin a release or set `revision` for a reproducible study. The live track's rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation; a static split says nothing about data that arrive later.
- **Gap filling.** Unobserved cells are forward-filled; only a series' leading gap is back-filled from its first reading. The loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training; fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Loaders keep the last partial batch for every split.

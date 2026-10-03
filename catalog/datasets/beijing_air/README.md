---
name: "beijing_air"
description: "Custom-loader preset for a flat Beijing air-quality CSV whose origin, statistics, and license are not recorded. Use for running your own wide Beijing air-quality CSV with an `OT` target; not for published or comparable results (use aqshunyi or aqwan, the documented UCI stations)."
---

# beijing_air

## Overview

`beijing_air` is a generic custom-loader preset for `dataset/BeijingAirQuality/BeijingAirQuality.csv`. The repository does not ship the file, does not pin its source, and records no statistics, so this card documents the contract only: a CSV with a `date` column, numeric channels, and an `OT` target. For documented Beijing air-quality data use [`aqshunyi`](../aqshunyi/README.md) or [`aqwan`](../aqwan/README.md), which come from the UCI Beijing Multi-Site data via TFB.

- **Provenance and license.** No source, citation, or license is recorded; `license` and `redistribution` are `unknown`, so do not publish the file.
- **Statistics.** Length, channel count, frequency, span, and missing values are unknown until `tsf data inspect --config configs/datasets/beijing_air.toml` runs against a real copy. When you adopt the file, record its origin, license, span, and statistics here and set `stats_basis` to `measured`.

## Protocol and pitfalls

- Preset defaults: `target = "OT"`, `scale = true`, `split_ratio = [0.7, 0.1, 0.2]`; scaling statistics come from the training split only. Lookback 96, horizons 96/192/336/720 (generic LTSF defaults; no published protocol).
- The target column must be named `OT`, the first column must be a parseable `date`, and every other column must be numeric.
- **Target column.** Unlike `aqshunyi`/`aqwan` (target `WSPM`), this preset keeps `OT` because its file is neither shipped nor pinned, so its real column names are unverifiable here; change `target` to match your file. `features = "S"` forecasts `OT`; `"MS"` forecasts the last column.
- **`drop_last`.** Loaders keep the last partial batch for every split.

---
name: "beijing_air"
kind: "dataset"
summary: "Preset for a flat Beijing air-quality CSV whose origin, statistics, and license are not recorded in the repository; treat every fact about the file as unverified until it is inspected."
domain: "Environment / air quality"
tags: ["air-quality", "beijing", "pollution", "environment", "csv", "unverified", "custom-loader"]
source: "not recorded in the repository (user-supplied CSV)"
source_url: "n/a"
citation: "n/a"
citation_url: "n/a"
license: "unknown"
redistribution: "unknown"
frequency: "unknown (not recorded)"
time_span: "unknown (file not pinned)"
length: "unknown (file not pinned)"
channels: "unknown (file not pinned)"
channel_kind: "channels"
target: "OT (the column the preset names; must exist in the file)"
missing_values: "unknown"
protocol: "TSFLab: chronological 7:1:2 split, scaling fitted on the training split only; lookback 96, horizons 96/192/336/720"
literature_protocol: "No published protocol; generic LTSF defaults (lookback 96, horizons 96/192/336/720, chronological 7:1:2) apply"
seq_lens: [96]
pred_lens: [96, 192, 336, 720]
split: "7:1:2"
stats_basis: "source-reported"
related: ["aqshunyi", "aqwan"]
config: "configs/datasets/beijing_air.toml"
loader: "custom"
alias: "beijing_air"
task_modes: ["time_series"]
---

# beijing_air

## Overview

`beijing_air` is a generic custom-loader preset for `dataset/BeijingAirQuality/BeijingAirQuality.csv`. The repository does not ship the file, does not pin its source, and records no statistics, so this card documents the contract only: a CSV with a `date` column, numeric channels, and an `OT` target. For documented Beijing air-quality data use `aqshunyi` or `aqwan`, which come from the UCI Beijing Multi-Site data.

## Provenance and license

- No source, citation, or license is recorded for this file. `license` and `redistribution` are `unknown`; do not publish it.
- When you adopt it, replace this card's facts after inspecting the file: record its origin, license, span, and statistics and set `stats_basis` to `measured`.

## Structure and statistics

No source-reported or measured statistics exist for this file: length, channel count, frequency, span, and missing values are all unknown until `tsf data inspect --config configs/datasets/beijing_air.toml` runs against a real copy.

## Standard protocol and known pitfalls

- Preset defaults: `target = "OT"`, `scale = true`, `split_ratio = [0.7, 0.1, 0.2]`; scaling statistics come from the training split only.
- The target column must be named `OT`, the first column must be a parseable `date`, and every other column must be numeric.
- **`drop_last`.** Loaders keep the last partial batch for every split.
- **Target column.** Unlike `aqshunyi`/`aqwan` (target `WSPM`), this preset keeps `OT` because its file is neither shipped nor pinned, so its real column names are unverifiable here; change `target` to match your file. `features = "S"` forecasts `OT`; `"MS"` forecasts the last column.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `custom`
- Config: [`configs/datasets/beijing_air.toml`](../../../configs/datasets/beijing_air.toml)
- Local path: `./dataset/BeijingAirQuality/BeijingAirQuality.csv`
- Dataset id: `(not applicable)`
- Track: `standard`

## Input and output contract

Each item provides history/target windows and timestamp marks `(year, month, day, weekday, hour, minute)` parsed from the file's date column; after batching, values use `[batch, time, channels]`.

Sequence length, label length, feature mode, and batch size are supplied by the
experiment task unless explicitly overridden below.

## Dataset parameters

```json
{
  "scale": true,
  "split_ratio": [
    0.7,
    0.1,
    0.2
  ],
  "target": "OT"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf data inspect --config configs/datasets/beijing_air.toml`. Not published and not downloadable with TSFLab: place the data file at the local path above (`./dataset/BeijingAirQuality/BeijingAirQuality.csv`) yourself; `tsf data prepare` has no converter for it. Reference this preset from an experiment configuration rather than duplicating its loader parameters.

## Composition constraints

Task modes: `time_series`. Match one of them to the model's declared task mode; the
config loader rejects other combinations. Change feature or scaling parameters
only after reading the loader. Paths are repository defaults and may need local
overrides; the card does not imply that the data is bundled.
<!-- dataset-card:canonical:end -->

## Related datasets

- [`aqshunyi`](../aqshunyi/README.md): documented Beijing station data (UCI via TFB)
- [`aqwan`](../aqwan/README.md): documented Beijing station data (UCI via TFB)

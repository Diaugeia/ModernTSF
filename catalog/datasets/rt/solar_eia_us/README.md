---
name: "rt/solar_eia_us"
kind: "dataset"
summary: "Hourly utility-scale solar generation per US balancing authority from EIA-930; frozen release of the real-time track solar_eia_us, seq_len 168 and horizon 24."
domain: "Energy / solar"
tags: ["solar", "generation", "eia-930", "balancing-authority", "united-states", "hourly", "real-time", "spatiotemporal"]
source: "EIA-930 hourly utility-scale solar generation per US balancing authority"
source_url: "https://www.eia.gov/electricity/gridmonitor/"
citation: "n/a (no dataset paper; TSFLab real-time track)"
citation_url: "n/a"
license: "US government work, public domain (per docs/en/realtime.md)"
redistribution: "allowed"
frequency: "hourly (1h)"
time_span: "365 days back from the bootstrap; grows with each weekly release"
length: "grows weekly; the bootstrap covers 365 days of API backfill"
channels: "unknown (balancing authorities reporting at least the configured share of hours at bootstrap; the set is in the store manifest)"
channel_kind: "stations"
target: "solar generation (megawatt-hours per hour per the EIA-930 series; unit not re-measured) per balancing authority"
missing_values: "operator-reported values have gaps and outliers; unobserved hours are NaN and the loader forward-fills them"
protocol: "TSFLab static protocol on a frozen release: chronological 7:1:2 split, one z-score from the training rows, seq_len 168 and horizon 24 as in the real-time track; the live rolling rounds are the separate real-time evaluation"
seq_lens: [168]
pred_lens: [24]
split: "7:1:2"
stats_basis: "source-reported"
related: ["rt/grid_eia_us", "rt/solar_openmeteo_ghi", "solar"]
realtime_track: "solar_eia_us"
config: "configs/datasets/rt/solar_eia_us.toml"
loader: "realtime_panel_st"
alias: "rt_solar_eia_us"
task_modes: ["spatiotemporal", "covariate"]
---

# rt/solar_eia_us

## Overview

`rt_solar_eia_us` serves hourly utility-scale solar generation per balancing authority as a static spatiotemporal dataset.

## Provenance and license

- Source: the EIA-930 Hourly Electric Grid Monitor (https://www.eia.gov/electricity/gridmonitor/) through the EIA Open Data API (free key). `docs/en/realtime.md` records US government public-domain terms.
- Releases are mirrored to the Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`solar_eia_us/`).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Bootstrap | 365 days of API backfill | source-reported (config) |
| Track config | `configs/realtime/solar_eia_us.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.8, timezone UTC | source-reported (config) |

No local copy of `dataset/realtime/solar_eia_us` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

## Standard protocol and known pitfalls

- **Operator-reported.** A few outliers and gaps are normal; regional aggregates such as `US48` or `TEX` are excluded so channels do not overlap.
- **Night zeros and lag.** Generation is zero at night, and the live feed lags about half a day.

- **Frozen snapshot, not the live track.** The preset reads the local panel store at a fixed release (`version`); the live track keeps growing and its rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation. A static split of the snapshot says nothing about data that arrive after it.
- **Gap filling.** Unobserved cells are forward-filled and only a series' leading gap is back-filled from its first reading; the loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training. Fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `realtime_panel_st`
- Config: [`configs/datasets/rt/solar_eia_us.toml`](../../../../configs/datasets/rt/solar_eia_us.toml)
- Local path: `./dataset/realtime/solar_eia_us`
- Dataset id: `(not applicable)`
- Track: `realtime`

## Input and output contract

Each item is `(value_history, value_future, covariate_history, covariate_future)`; values use `[time, nodes]` and covariates `[time, nodes, features]` before batching.

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
  "track": "solar_eia_us"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/rt/solar_eia_us.toml`; fetch
published files with `tsf dataset download rt/solar_eia_us` when the preset is
listed by `tsf dataset download --list`, otherwise place the data at the local
path above (see `tsf dataset prepare --help`). Reference this preset from an
experiment configuration rather than duplicating its loader parameters.

## Composition constraints

Task modes: `spatiotemporal`, `covariate`. Match one of them to the model's declared task mode; the
config loader rejects other combinations. Change feature or scaling parameters
only after reading the loader. Paths are repository defaults and may need local
overrides; the card does not imply that the data is bundled.
<!-- dataset-card:canonical:end -->

## Related datasets

- [`rt/grid_eia_us`](../grid_eia_us/README.md): same source, demand
- [`rt/solar_openmeteo_ghi`](../solar_openmeteo_ghi/README.md): irradiance driver
- [`solar`](../../solar/README.md): static PV power set

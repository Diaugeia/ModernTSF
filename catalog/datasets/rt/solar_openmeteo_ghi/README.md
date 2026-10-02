---
name: "rt/solar_openmeteo_ghi"
kind: "dataset"
summary: "Hourly global horizontal irradiance at 55 PV-relevant sites from Open-Meteo; frozen release of the real-time track solar_openmeteo_ghi, seq_len 168 and horizon 24."
domain: "Energy / solar"
tags: ["solar", "irradiance", "ghi", "open-meteo", "hourly", "real-time", "spatiotemporal", "covariate"]
source: "Open-Meteo Historical Forecast API, shortwave radiation (global horizontal irradiance)"
source_url: "https://open-meteo.com/en/docs/historical-forecast-api"
citation: "n/a (no dataset paper; TSFLab real-time track)"
citation_url: "n/a"
license: "CC-BY-4.0 (Open-Meteo; non-commercial free tier, attribution required)"
redistribution: "conditional"
frequency: "hourly (1h)"
time_span: "the 365 days before the bootstrap; grows with each weekly release"
length: "grows weekly; the bootstrap covers 365 days of the same API"
channels: "55 (sites in the track config)"
channel_kind: "stations"
target: "global horizontal irradiance per site (watts per square metre per the API default; unit not re-measured)"
missing_values: "model-analysis grid values have no gaps by construction in normal operation; any unobserved hours are NaN in the store and forward-filled"
protocol: "TSFLab static protocol on a frozen release: chronological 7:1:2 split, one z-score from the training rows, seq_len 168 and horizon 24 as in the real-time track; the live rolling rounds are the separate real-time evaluation"
seq_lens: [168]
pred_lens: [24]
split: "7:1:2"
stats_basis: "source-reported"
related: ["rt/solar_eia_us", "solar"]
realtime_track: "solar_openmeteo_ghi"
config: "configs/datasets/rt/solar_openmeteo_ghi.toml"
loader: "realtime_panel_st"
alias: "rt_solar_openmeteo_ghi"
task_modes: ["spatiotemporal", "covariate"]
---

# rt/solar_openmeteo_ghi

## Overview

`rt_solar_openmeteo_ghi` serves hourly global horizontal irradiance at 55 sites as a static spatiotemporal dataset; irradiance is the weather driver of PV output and needs no API key.

## Provenance and license

- Source: the Open-Meteo Historical Forecast API (https://open-meteo.com), which stitches the analysis hours of operational weather models. `docs/en/realtime.md` records CC BY 4.0, a non-commercial free tier, and attribution.
- Releases are mirrored to the Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`solar_openmeteo_ghi/`).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Bootstrap | 365 days of the same API | source-reported (config) |
| Track config | `configs/realtime/solar_openmeteo_ghi.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.9, timezone UTC | source-reported (config) |

No local copy of `dataset/realtime/solar_openmeteo_ghi` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

## Standard protocol and known pitfalls

- **Model values, not station readings.** Values are model analysis at coordinates, so they are smoother than gauge data; the newest three hours are forecasts and are not stored.
- **Night zeros.** Irradiance is exactly zero at night, so MAPE-style metrics are undefined and MSE is driven by daytime peaks.
- **Frozen snapshot, not the live track.** No `version` is pinned, so the preset reads whatever the local panel store holds (pin a release, or set `revision`, for a reproducible study); the live track keeps growing and its rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation. A static split of the snapshot says nothing about data that arrive after it.
- **Gap filling.** Unobserved cells are forward-filled and only a series' leading gap is back-filled from its first reading; the loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training. Fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `realtime_panel_st`
- Config: [`configs/datasets/rt/solar_openmeteo_ghi.toml`](../../../../configs/datasets/rt/solar_openmeteo_ghi.toml)
- Local path: `./dataset/realtime/solar_openmeteo_ghi`
- Dataset id: `(not applicable)`
- Track: `realtime`

## Input and output contract

Each item is `(value_history, value_future, covariate_history, covariate_future)`; values use `[time, nodes]` and covariates `[time, nodes, features]` before batching. There are no separate timestamp marks: calendar information, when present, is carried by the covariates.

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
  "track": "solar_openmeteo_ghi"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf data inspect --config configs/datasets/rt/solar_openmeteo_ghi.toml`. Not published as files. The panel store is written by the real-time track (see `docs/en/realtime.md` and `tsf realtime --help`) under the local path above, or pulled from a Hub revision via `dataset.params.revision`. Reference this preset from an experiment configuration rather than duplicating its loader parameters.

## Composition constraints

Task modes: `spatiotemporal`, `covariate`. Match one of them to the model's declared task mode; the
config loader rejects other combinations. Change feature or scaling parameters
only after reading the loader. Paths are repository defaults and may need local
overrides; the card does not imply that the data is bundled.
<!-- dataset-card:canonical:end -->

## Related datasets

- [`rt/solar_eia_us`](../solar_eia_us/README.md): US PV generation as a real-time preset
- [`solar`](../../solar/README.md): static PV power set

---
name: "rt/weather_openmeteo_temp"
kind: "dataset"
summary: "Hourly 2 m temperature at 82 US and EU cities from the Open-Meteo Historical Forecast API; frozen release of the real-time track weather_openmeteo_temp, seq_len 168 and horizon 24."
domain: "Weather / temperature"
tags: ["weather", "temperature", "open-meteo", "hourly", "cities", "real-time", "spatiotemporal", "covariate"]
source: "Open-Meteo Historical Forecast API, 2 m temperature"
source_url: "https://open-meteo.com/en/docs/historical-forecast-api"
citation: "n/a (no dataset paper; TSFLab real-time track)"
citation_url: "n/a"
license: "CC-BY-4.0 (Open-Meteo; non-commercial free tier, attribution required)"
redistribution: "conditional"
frequency: "hourly (1h)"
time_span: "from 2022 model-analysis data back 365 days at bootstrap; grows with each weekly release"
length: "grows weekly; the bootstrap covers 365 days of the same API (analysis hours from 2022)"
channels: "82 (cities in the track config)"
channel_kind: "stations"
target: "2 m air temperature per city (degrees Celsius per the API default; unit not re-measured)"
missing_values: "model-analysis grid values have no gaps by construction in normal operation; any unobserved hours are NaN in the store and forward-filled"
protocol: "TSFLab static protocol on a frozen release: chronological 7:1:2 split, one z-score from the training rows, seq_len 168 and horizon 24 as in the real-time track; the live rolling rounds are the separate real-time evaluation"
seq_lens: [168]
pred_lens: [24]
split: "7:1:2"
stats_basis: "source-reported"
related: ["rt/solar_openmeteo_ghi", "weather"]
realtime_track: "weather_openmeteo_temp"
config: "configs/datasets/rt/weather_openmeteo_temp.toml"
loader: "realtime_panel_st"
alias: "rt_weather_openmeteo_temp"
task_modes: ["spatiotemporal", "covariate"]
---

# rt/weather_openmeteo_temp

## Overview

`rt_weather_openmeteo_temp` serves hourly temperature at 82 cities as a static spatiotemporal dataset (cities as nodes plus calendar covariates), read from the panel store at a fixed release.

## Provenance and license

- Source: the Open-Meteo Historical Forecast API (https://open-meteo.com), which stitches the analysis hours of operational weather models. `docs/en/realtime.md` records CC BY 4.0, a non-commercial free tier, and attribution.
- Releases are mirrored to the Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`weather_openmeteo_temp/`).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Bootstrap | 365 days of the same API (analysis hours from 2022) | source-reported (config) |
| Track config | `configs/realtime/weather_openmeteo_temp.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.9, timezone UTC | source-reported (config) |

No local copy of `dataset/realtime/weather_openmeteo_temp` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

## Standard protocol and known pitfalls

- **Model values, not station readings.** Values are model analysis at coordinates, so they are smoother than gauge data; the newest three hours are forecasts and are not stored.
- **Strong annual cycle.** A one-year history gives the training split one season and the test split another, so the seasonal shift is large.

- **Frozen snapshot, not the live track.** The preset reads the local panel store at a fixed release (`version`); the live track keeps growing and its rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation. A static split of the snapshot says nothing about data that arrive after it.
- **Gap filling.** Unobserved cells are forward-filled and only a series' leading gap is back-filled from its first reading; the loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training. Fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `realtime_panel_st`
- Config: [`configs/datasets/rt/weather_openmeteo_temp.toml`](../../../../configs/datasets/rt/weather_openmeteo_temp.toml)
- Local path: `./dataset/realtime/weather_openmeteo_temp`
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
  "track": "weather_openmeteo_temp"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf data inspect --config configs/datasets/rt/weather_openmeteo_temp.toml`; fetch
published files with `tsf data download rt/weather_openmeteo_temp` when the preset is
listed by `tsf data download --list`, otherwise place the data at the local
path above (see `tsf data prepare --help`). Reference this preset from an
experiment configuration rather than duplicating its loader parameters.

## Composition constraints

Task modes: `spatiotemporal`, `covariate`. Match one of them to the model's declared task mode; the
config loader rejects other combinations. Change feature or scaling parameters
only after reading the loader. Paths are repository defaults and may need local
overrides; the card does not imply that the data is bundled.
<!-- dataset-card:canonical:end -->

## Related datasets

- [`rt/solar_openmeteo_ghi`](../solar_openmeteo_ghi/README.md): same source, irradiance
- [`weather`](../../weather/README.md): static single-station weather set

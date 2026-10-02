---
name: "rt/air_openaq_us"
kind: "dataset"
summary: "Hourly PM2.5 from US reference monitors aggregated by OpenAQ v3, up to 200 stations as nodes; frozen release of the real-time track air_openaq_us, seq_len 168 and horizon 24."
domain: "Environment / air quality"
tags: ["air-quality", "pm25", "openaq", "united-states", "hourly", "real-time", "stations", "spatiotemporal", "covariate"]
source: "OpenAQ v3 hourly averages of PM2.5, US government reference monitors only"
source_url: "https://openaq.org"
citation: "n/a (no dataset paper; TSFLab real-time track)"
citation_url: "n/a"
license: "CC-BY-4.0 (OpenAQ; per-provider licenses vary)"
redistribution: "conditional"
frequency: "hourly (1h)"
time_span: "from the first bootstrap week back 365 days; grows with each weekly release"
length: "grows weekly; the bootstrap covers 365 days of API backfill"
channels: "up to 200 (config max_sensors; the exact set is in the store manifest)"
channel_kind: "stations"
target: "hourly PM2.5 (micrograms per cubic metre) per station"
missing_values: "stations report irregularly; unobserved hours are NaN in the store and the loader forward-fills them"
protocol: "TSFLab static protocol on a frozen release: chronological 7:1:2 split, one z-score from the training rows, seq_len 168 and horizon 24 as in the real-time track; the live rolling rounds are the separate real-time evaluation"
seq_lens: [168]
pred_lens: [24]
split: "7:1:2"
stats_basis: "source-reported"
related: ["rt/air_openaq_cn", "rt/air_airnow_us", "rt/air_openaq_eu"]
realtime_track: "air_openaq_us"
config: "configs/datasets/rt/air_openaq_us.toml"
loader: "realtime_panel_st"
alias: "rt_air_openaq_us"
task_modes: ["spatiotemporal", "covariate"]
---

# rt/air_openaq_us

## Overview

`rt_air_openaq_us` serves the OpenAQ US PM2.5 real-time track as a static spatiotemporal dataset; low-cost sensor networks are excluded (`monitor_only`).

## Provenance and license

- Source: OpenAQ v3 (https://openaq.org), reference monitors only; an API key is needed to fetch updates. Terms are CC BY 4.0 with per-provider licenses (`docs/en/realtime.md`), hence `conditional`.
- Releases are mirrored to the Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`air_openaq_us/`).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Bootstrap | 365 days of API backfill | source-reported (config) |
| Track config | `configs/realtime/air_openaq_us.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.7, timezone UTC | source-reported (config) |

No local copy of `dataset/realtime/air_openaq_us` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

## Standard protocol and known pitfalls

- **Sensor sets are fixed at bootstrap.** The channel set is frozen so rounds stay comparable; stations that start reporting later are not added, and stations that stop remain as forward-filled constants.
- **Reported values are noisy.** PM2.5 readings can be sparse, spiky, and include instrument noise; the real-time store keeps them as reported.

- **Frozen snapshot, not the live track.** The preset reads the local panel store at a fixed release (`version`); the live track keeps growing and its rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation. A static split of the snapshot says nothing about data that arrive after it.
- **Gap filling.** Unobserved cells are forward-filled and only a series' leading gap is back-filled from its first reading; the loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training. Fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `realtime_panel_st`
- Config: [`configs/datasets/rt/air_openaq_us.toml`](../../../../configs/datasets/rt/air_openaq_us.toml)
- Local path: `./dataset/realtime/air_openaq_us`
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
  "track": "air_openaq_us"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/rt/air_openaq_us.toml`; fetch
published files with `tsf dataset download rt/air_openaq_us` when the preset is
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

- [`rt/air_openaq_cn`](../air_openaq_cn/README.md): China, same source
- [`rt/air_airnow_us`](../air_airnow_us/README.md): US, EPA AirNow source
- [`rt/air_openaq_eu`](../air_openaq_eu/README.md): Europe, same source

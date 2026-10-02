---
name: "rt/air_openaq_eu"
kind: "dataset"
summary: "Hourly PM2.5 from reference monitors in DE, FR, ES, IT, PL, GB, NL, AT, CZ and SE via OpenAQ v3, up to 200 stations as nodes; frozen release of the real-time track air_openaq_eu."
domain: "Environment / air quality"
tags: ["air-quality", "pm25", "openaq", "europe", "hourly", "real-time", "stations", "spatiotemporal", "covariate"]
source: "OpenAQ v3 hourly averages of PM2.5, government reference monitors in ten European countries"
source_url: "https://openaq.org"
citation: "n/a (no dataset paper; TSFLab real-time track)"
citation_url: "n/a"
license: "CC-BY-4.0 (OpenAQ; per-provider licenses vary)"
redistribution: "conditional"
frequency: "hourly (1h)"
time_span: "the 365 days before the bootstrap; grows with each weekly release"
length: "grows weekly; the bootstrap covers 365 days of API backfill"
channels: "up to 200 (config max_sensors, spread over the listed countries; the exact set is in the store manifest)"
channel_kind: "stations"
target: "hourly PM2.5 (micrograms per cubic metre) per station"
missing_values: "stations report irregularly; unobserved hours are NaN in the store and the loader forward-fills them"
protocol: "TSFLab static protocol on a frozen release: chronological 7:1:2 split, one z-score from the training rows, seq_len 168 and horizon 24 as in the real-time track; the live rolling rounds are the separate real-time evaluation"
seq_lens: [168]
pred_lens: [24]
split: "7:1:2"
stats_basis: "source-reported"
related: ["rt/air_openaq_us", "rt/air_openaq_cn"]
realtime_track: "air_openaq_eu"
config: "configs/datasets/rt/air_openaq_eu.toml"
loader: "realtime_panel_st"
alias: "rt_air_openaq_eu"
task_modes: ["spatiotemporal", "covariate"]
---

# rt/air_openaq_eu

## Overview

`rt_air_openaq_eu` serves the OpenAQ Europe PM2.5 real-time track as a static spatiotemporal dataset; sensors are spread evenly over the ten listed countries.

## Provenance and license

- Source: OpenAQ v3 (https://openaq.org), reference monitors only; an API key is needed to fetch updates. Terms are CC BY 4.0 with per-provider licenses (`docs/en/realtime.md`), hence `conditional`.
- Releases are mirrored to the Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`air_openaq_eu/`).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Bootstrap | 365 days of API backfill | source-reported (config) |
| Track config | `configs/realtime/air_openaq_eu.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.7, timezone UTC | source-reported (config) |

No local copy of `dataset/realtime/air_openaq_eu` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

## Standard protocol and known pitfalls

- **Sensor sets are fixed at bootstrap.** The channel set is frozen so rounds stay comparable; stations that start reporting later are not added, and stations that stop remain as forward-filled constants.
- **Reported values are noisy.** PM2.5 readings can be sparse, spiky, and include instrument noise; the real-time store keeps them as reported.
- **Mixed time zones.** Stamps are naive UTC while stations span several European time zones, so local daily cycles are phase-shifted between channels.
- **Frozen snapshot, not the live track.** No `version` is pinned, so the preset reads whatever the local panel store holds (pin a release, or set `revision`, for a reproducible study); the live track keeps growing and its rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation. A static split of the snapshot says nothing about data that arrive after it.
- **Gap filling.** Unobserved cells are forward-filled and only a series' leading gap is back-filled from its first reading; the loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training. Fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `realtime_panel_st`
- Config: [`configs/datasets/rt/air_openaq_eu.toml`](../../../../configs/datasets/rt/air_openaq_eu.toml)
- Local path: `./dataset/realtime/air_openaq_eu`
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
  "track": "air_openaq_eu"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf data inspect --config configs/datasets/rt/air_openaq_eu.toml`; fetch
published files with `tsf data download rt/air_openaq_eu` when the preset is
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

- [`rt/air_openaq_us`](../air_openaq_us/README.md): US, same source
- [`rt/air_openaq_cn`](../air_openaq_cn/README.md): China, same source

---
name: "ultratraffic_sb_ts"
kind: "dataset"
summary: "Hourly total flow per Caltrans PeMS station in District 8 (San Bernardino / Riverside) for 2023, 1105 stations (all stations present in 2023), with stations flattened into channels."
domain: "Transport / road traffic"
tags: ["traffic", "pems", "flow", "hourly", "california", "district-8", "ultratraffic", "stations", "2023", "real-time-backed", "time-series", "high-dimensional"]
source: "Caltrans PeMS hourly total flow per station, as packaged in the UltraTraffic_CL archive (archive publisher not identified in the repository)"
source_url: "http://pems.dot.ca.gov"
citation: "n/a (no paper identified for the UltraTraffic_CL archive)"
citation_url: "n/a"
license: "unknown"
redistribution: "unknown"
frequency: "hourly (1h)"
time_span: "2023-01-01 00:00 to 2023-12-31 23:00 (store holds 2003-2023)"
length: 8760
channels: 1105
channel_kind: "channels"
target: "total flow of every station"
missing_values: "none after the converter's gap filling (measured: 0 NaN in the 2023 panel); zeros remain"
protocol: "Chronological 7:1:2 within 2023, global z-score from the training split; no published protocol (seq_len 168 and horizon 24 match the real-time track)"
seq_lens: [168]
pred_lens: [24]
split: "7:1:2"
stats_basis: "measured"
related: ["ultratraffic_sb_st", "ultratraffic_ba_st", "pems08"]
realtime_track: "traffic_pems_sb"
config: "configs/datasets/ultratraffic_sb_ts.toml"
loader: "ultratraffic_ts"
alias: "ultratraffic_sb_ts"
task_modes: ["time_series"]
---

# ultratraffic_sb_ts

## Overview

This preset loads one year (2023) of hourly total flow, in vehicles per hour summed over lanes, for the stations of Caltrans PeMS District 8 (San Bernardino / Riverside) from the local UltraTraffic parquet store (`dataset/ultratraffic/PEMS_SB`, built from `UltraTraffic_CL.zip`). All stations present in 2023; stations flattened into channels. The same store bootstraps the real-time track `traffic_pems_sb`.

## Provenance and license

- Source system: Caltrans PeMS (http://pems.dot.ca.gov), District 8. Its Conditions of Use say site information is in the public domain unless otherwise indicated; this is a generic policy.
- Packaging: the store manifest records `UltraTraffic_CL.zip` as its source archive. No publication or repository named UltraTraffic was found in primary sources during this card's research; the closest published relatives (XXLTraffic, CC BY-NC 4.0; TrafficStream) were not confirmed to be the same data, so provenance beyond PeMS is unverified. If the archive derives from a CC BY-NC dataset, commercial use and republication are restricted.
- Checked for terms: the PeMS Conditions of Use (https://pems.dot.ca.gov/?view=tou) say information is "in the public domain" unless otherwise indicated, which covers the raw PeMS data only. The `UltraTraffic_CL.zip` archive is not documented anywhere that was found (no paper, repository, or dataset card), so the packager's terms and possible upstream derivation (for example from XXLTraffic, CC BY-NC 4.0) cannot be verified. `license` and `redistribution` stay `unknown`; do not run `tsf data publish` for these presets without confirming the archive's terms.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Stations (static 2023) | 1,105 (this preset: 1,105) | measured |
| Rows (hours in 2023) | 8,760 | measured |
| Value range, 2023 (all stations) | 0 to 16,278, mean 2,442.2 | measured |
| Exact zeros, 2023 (all stations) | 8.31% of all values | measured |
| NaN, 2023 | 0.0% | measured |
| 2023 added / common stations (continual-learning split) | 51 added, 1,054 common | measured (manifest) |
| Stations per static year (2003-2023) | 108 to 1,252 | measured (manifest) |
| Real-time store `dataset/realtime/traffic_pems_sb` | 43,824 hourly rows, 2019-01-01 to 2023-12-31, 1,105 channels, 4.5% NaN (bootstrap release only) | measured |

Measured from the local parquet store and manifest under `dataset/` (read-only).

## Standard protocol and known pitfalls

- **One year only.** The preset loads 2023 alone: a 7:1:2 split puts training from January to mid-September (6,132 hours), validation to about 20 October, and the last 1,752 hours (late October to December, with Thanksgiving and Christmas) in test, so the test period includes holiday regimes the training split lacks.
- **Scaling.** One scalar mean and standard deviation is computed on the training rows across all stations (not per station), so large stations dominate and per-station scale differences remain; the loader's `value_mean`/`value_std` are used for inversion.
- **Gap filling.** The loader linearly interpolates (both directions) and zero-fills the whole panel before splitting; for the static 2023 panel there are no gaps, but other variants or years could use future values to fill earlier gaps.
- **Zeros.** Some stations report exact zeros for many hours (8% of the District 8 values, 4% of District 3), probably outages or closures rather than real empty roads; this was not traced to the source.
- **No adjacency.** The archive carries no station coordinates, so graph models receive no adjacency (`adj_mx` is None).
- **Different from `traffic`.** This is hourly flow in vehicles per hour for 2023, not LTSF's 2015-2016 occupancy rate.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `ultratraffic_ts`
- Config: [`configs/datasets/ultratraffic_sb_ts.toml`](../../../configs/datasets/ultratraffic_sb_ts.toml)
- Local path: `./dataset/ultratraffic`
- Dataset id: `(not applicable)`
- Track: `standard`

## Input and output contract

Each item contains history/future values shaped `[time, nodes]` (nodes become channels) plus timestamp marks before batching; covariates are dropped.

Sequence length, label length, feature mode, and batch size are supplied by the
experiment task unless explicitly overridden below.

## Dataset parameters

```json
{
  "region": "PEMS_SB",
  "scale": true,
  "split_ratio": [
    0.7,
    0.1,
    0.2
  ],
  "variant": "static",
  "years": [
    2023
  ]
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf data inspect --config configs/datasets/ultratraffic_sb_ts.toml`; fetch
published files with `tsf data download ultratraffic_sb_ts` when the preset is
listed by `tsf data download --list`, otherwise place the data at the local
path above (see `tsf data prepare --help`). Reference this preset from an
experiment configuration rather than duplicating its loader parameters.

## Composition constraints

Task modes: `time_series`. Match one of them to the model's declared task mode; the
config loader rejects other combinations. Change feature or scaling parameters
only after reading the loader. Paths are repository defaults and may need local
overrides; the card does not imply that the data is bundled.
<!-- dataset-card:canonical:end -->

## Related datasets

- [`ultratraffic_sb_st`](../ultratraffic_sb_st/README.md): same stations, other layout
- [`ultratraffic_ba_st`](../ultratraffic_ba_st/README.md): another district, same pipeline
- [`pems08`](../pems08/README.md): 5-minute PeMS flow graph of the same district (STSGCN)

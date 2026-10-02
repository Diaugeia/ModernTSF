---
name: "rt/traffic_pems_sb"
kind: "dataset"
summary: "Hourly total flow per Caltrans PeMS station in District 8 (San Bernardino and Riverside), 2019-2023, 1105 stations as nodes with calendar covariates; the frozen release of the real-time track traffic_pems_sb."
domain: "Transport / road traffic"
tags: ["traffic", "pems", "flow", "hourly", "california", "district-8", "real-time", "stations", "spatiotemporal", "frozen-release", "covariate"]
source: "Caltrans PeMS hourly total flow per station (District 8 (San Bernardino and Riverside)); 2019-2023 history from the UltraTraffic_CL archive, later weeks from the PeMS clearinghouse station_5min files"
source_url: "http://pems.dot.ca.gov"
citation: "n/a (no dataset paper; TSFLab real-time track)"
citation_url: "n/a"
license: "unknown"
redistribution: "unknown"
frequency: "hourly (1h)"
time_span: "2019-01-01 00:00 to 2023-12-31 23:00 in release 2026.09.28-1933 (grows with each weekly release)"
length: 43824
channels: 1105
channel_kind: "nodes"
target: "total flow of every station"
missing_values: "4.54% of cells are NaN in the stored panel (measured); the loader forward-fills them, back-filling only a series' start"
protocol: "TSFLab static protocol on a frozen release: chronological 7:1:2 split, one z-score from the training rows, seq_len 168 and horizon 24 as in the real-time track; the live rolling rounds are the separate real-time evaluation"
seq_lens: [168]
pred_lens: [24]
split: "7:1:2"
stats_basis: "measured"
related: ["ultratraffic_sb_st", "ultratraffic_sb_ts", "rt/traffic_pems_ba", "rt/traffic_pems_la"]
realtime_track: "traffic_pems_sb"
config: "configs/datasets/rt/traffic_pems_sb.toml"
loader: "realtime_panel_st"
alias: "rt_traffic_pems_sb"
task_modes: ["spatiotemporal", "covariate"]
---

# rt/traffic_pems_sb

## Overview

`rt_traffic_pems_sb` serves the hourly total vehicle flow of every station in Caltrans PeMS District 8 (San Bernardino and Riverside) as a static dataset: the real-time panel store `dataset/realtime/traffic_pems_sb` is read at release `2026.09.28-1933` and exposed in the spatiotemporal layout (stations as nodes, plus time-of-day and day-of-week covariates). It is the same panel a real-time round uses as history, so a model trained here can be compared with the live `traffic_pems_sb` track.

## Provenance and license

- Source system: Caltrans PeMS (http://pems.dot.ca.gov), District 8. Its Conditions of Use say site information is in the public domain unless otherwise indicated; this is a generic policy, not a statement about the packaged data.
- The 2019-2023 bootstrap history comes from the UltraTraffic_CL archive (publisher not identified in the repository; see `ultratraffic_sb_st`); weekly increments come from the PeMS clearinghouse `station_5min` files summed to hourly flow (free PeMS account).
- `license` and `redistribution` are `unknown`; confirm the archive's terms before publishing a release or this preset's data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Release | `2026.09.28-1933` (bootstrap release; the manifest holds this one release) | measured |
| Rows | 43,824 hourly (2019-01-01 00:00 to 2023-12-31 23:00; a contiguous hourly grid) | measured |
| Stations | 1,105 | measured |
| NaN cells in the store | 4.54% (per year 2019-2023: 6.8, 6.7, 4.6, 4.6, 0.0 %) | measured |
| Stations with more than 50% NaN | 49 | measured |
| Stations whose first reading is after the training split ends | 47 | measured |
| Value range | 0 to 16,278; mean 2535.7, standard deviation 1674.4 (vehicles per hour, NaN excluded) | measured |
| Exact zeros | 1.94% of observed values | measured |
| Mean flow by year | 2,671 (2019), 2,427 (2020), 2,582 (2021), 2,562 (2022), 2,442 (2023) | measured |
| Track config | `configs/realtime/traffic_pems_sb.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.8, timezone America/Los_Angeles | source-reported (config) |

Measured from the local store `dataset/realtime/traffic_pems_sb` (read-only). With 7:1:2 the training rows end 2022-07-02 and the validation rows end 2022-12-31; the test split is therefore all of 2023, the complete-coverage year.

## Standard protocol and known pitfalls

- **Coverage is not uniform.** The NaN share falls from 6.8% in 2019 to 0.0% in 2023, so the test split is the best-covered year while the training split carries most of the filled gaps.
- **2020 regime.** Mean flow dips in 2020 and does not return to its 2019 level in every district (see the yearly means above), so the training split mixes pre-pandemic and pandemic regimes.
- **No adjacency.** The store carries no station coordinates, so graph models receive no adjacency.
- **Frozen snapshot, not the live track.** The preset reads the local panel store at a fixed release (`version`); the live track keeps growing and its rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation. A static split of the snapshot says nothing about data that arrive after it.
- **Gap filling.** Unobserved cells are forward-filled and only a series' leading gap is back-filled from its first reading; the loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training. Fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `realtime_panel_st`
- Config: [`configs/datasets/rt/traffic_pems_sb.toml`](../../../../configs/datasets/rt/traffic_pems_sb.toml)
- Local path: `./dataset/realtime/traffic_pems_sb`
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
  "track": "traffic_pems_sb",
  "version": "2026.09.28-1933"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/rt/traffic_pems_sb.toml`; fetch
published files with `tsf dataset download rt/traffic_pems_sb` when the preset is
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

- [`ultratraffic_sb_st`](../../ultratraffic_sb_st/README.md): the same district, 2023 only, loaded from the UltraTraffic store
- [`ultratraffic_sb_ts`](../../ultratraffic_sb_ts/README.md): the same data with stations as plain channels
- [`rt/traffic_pems_ba`](../traffic_pems_ba/README.md): another PeMS district as a real-time preset
- [`rt/traffic_pems_la`](../traffic_pems_la/README.md): another PeMS district as a real-time preset

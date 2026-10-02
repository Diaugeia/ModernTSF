---
name: "rt/traffic_pems_la"
kind: "dataset"
summary: "Hourly total flow per Caltrans PeMS station in District 7 (Los Angeles and Ventura), 2019-2023, 1926 stations as nodes with calendar covariates; the frozen release of the real-time track traffic_pems_la."
domain: "Transport / road traffic"
tags: ["traffic", "pems", "flow", "hourly", "california", "district-7", "real-time", "stations", "spatiotemporal", "frozen-release", "covariate"]
source: "Caltrans PeMS hourly total flow per station (District 7 (Los Angeles and Ventura)); 2019-2023 history from the UltraTraffic_CL archive, later weeks from the PeMS clearinghouse station_5min files"
source_url: "https://pems.dot.ca.gov"
citation: "n/a (no dataset paper; TSFLab real-time track)"
citation_url: "n/a"
license: "unknown"
redistribution: "unknown"
frequency: "hourly (1h)"
time_span: "2019-01-01 00:00 to 2023-12-31 23:00 in release 2026.09.28-2003 (grows with each weekly release)"
length: 43824
channels: 1926
channel_kind: "nodes"
target: "total flow of every station"
missing_values: "1.30% of cells are NaN in the stored panel (measured); the loader forward-fills them, back-filling only a series' start"
protocol: "TSFLab static protocol on a frozen release: chronological 7:1:2 split, one z-score from the training rows, seq_len 168 and horizon 24 as in the real-time track; the live rolling rounds are the separate real-time evaluation"
seq_lens: [168]
pred_lens: [24]
split: "7:1:2"
stats_basis: "measured"
related: ["ultratraffic_la_st", "ultratraffic_la_ts", "rt/traffic_pems_ba", "rt/traffic_pems_sac", "rt/traffic_pems_sb"]
realtime_track: "traffic_pems_la"
config: "configs/datasets/rt/traffic_pems_la.toml"
loader: "realtime_panel_st"
alias: "rt_traffic_pems_la"
task_modes: ["spatiotemporal", "covariate"]
---

# rt/traffic_pems_la

## Overview

`rt_traffic_pems_la` serves the hourly total vehicle flow of every station in Caltrans PeMS District 7 (Los Angeles and Ventura) as a static dataset: the real-time panel store `dataset/realtime/traffic_pems_la` is read at release `2026.09.28-2003` and exposed in the spatiotemporal layout (stations as nodes, plus time-of-day and day-of-week covariates). It is the same panel a real-time round uses as history, so a model trained here can be compared with the live `traffic_pems_la` track.

## Provenance and license

- Source system: Caltrans PeMS (https://pems.dot.ca.gov), District 7. Its Conditions of Use say site information is in the public domain unless otherwise indicated; this is a generic policy, not a statement about the packaged data.
- The 2019-2023 bootstrap history comes from the UltraTraffic_CL archive (publisher not identified in the repository; see `ultratraffic_la_st`); weekly increments come from the PeMS clearinghouse `station_5min` files summed to hourly flow (free PeMS account).
- `license` and `redistribution` are `unknown`; confirm the archive's terms before publishing a release or this preset's data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Release | `2026.09.28-2003` (bootstrap release; the manifest holds this one release) | measured |
| Rows | 43,824 hourly (2019-01-01 00:00 to 2023-12-31 23:00; a contiguous hourly grid) | measured |
| Stations | 1,926 | measured |
| NaN cells in the store | 1.30% (per year 2019-2023: 2.9, 2.0, 1.3, 0.3, 0.0 %) | measured |
| Stations with more than 50% NaN | 26 | measured |
| Stations whose first reading is after the training split ends | 5 | measured |
| Value range | 0 to 23,676; mean 3578.5, standard deviation 2242.1 (vehicles per hour, NaN excluded) | measured |
| Exact zeros | 0.38% of observed values | measured |
| Mean flow by year | 3,825 (2019), 3,357 (2020), 3,619 (2021), 3,568 (2022), 3,527 (2023) | measured |
| Track config | `configs/realtime/traffic_pems_la.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.8, timezone America/Los_Angeles | source-reported (config) |

Measured from the local store `dataset/realtime/traffic_pems_la` (read-only). With 7:1:2 the training rows end 2022-07-02 and the validation rows end 2022-12-31; the test split is therefore all of 2023, the complete-coverage year.

## Standard protocol and known pitfalls

- **Coverage is not uniform.** The NaN share falls from 2.9% in 2019 to 0.0% in 2023, so the test split is the best-covered year while the training split carries most of the filled gaps.
- **2020 regime.** Mean flow dips in 2020 and does not return to its 2019 level in every district (see the yearly means above), so the training split mixes pre-pandemic and pandemic regimes.
- **No adjacency.** The store carries no station coordinates, so graph models receive no adjacency.
- **Frozen snapshot, not the live track.** The preset reads the local panel store at the pinned release (`version`); the live track keeps growing and its rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation. A static split of the snapshot says nothing about data that arrive after it.
- **Gap filling.** Unobserved cells are forward-filled and only a series' leading gap is back-filled from its first reading; the loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training. Fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `realtime_panel_st`
- Config: [`configs/datasets/rt/traffic_pems_la.toml`](../../../../configs/datasets/rt/traffic_pems_la.toml)
- Local path: `./dataset/realtime/traffic_pems_la`
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
  "track": "traffic_pems_la",
  "version": "2026.09.28-2003"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf data inspect --config configs/datasets/rt/traffic_pems_la.toml`; fetch
published files with `tsf data download rt/traffic_pems_la` when the preset is
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

- [`ultratraffic_la_st`](../../ultratraffic_la_st/README.md): the same district, 2023 only, loaded from the UltraTraffic store
- [`ultratraffic_la_ts`](../../ultratraffic_la_ts/README.md): the same data with stations as plain channels
- [`rt/traffic_pems_ba`](../traffic_pems_ba/README.md): another PeMS district as a real-time preset
- [`rt/traffic_pems_sac`](../traffic_pems_sac/README.md): another PeMS district as a real-time preset
- [`rt/traffic_pems_sb`](../traffic_pems_sb/README.md): another PeMS district as a real-time preset

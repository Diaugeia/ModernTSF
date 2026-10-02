---
name: "rt/traffic_pems_ba"
kind: "dataset"
summary: "Hourly total flow per Caltrans PeMS station in District 4 (San Francisco Bay Area), 2019-2023, 2472 stations as nodes with calendar covariates; the frozen release of the real-time track traffic_pems_ba."
domain: "Transport / road traffic"
tags: ["traffic", "pems", "flow", "hourly", "california", "district-4", "real-time", "stations", "spatiotemporal", "frozen-release", "covariate"]
source: "Caltrans PeMS hourly total flow per station (District 4 (San Francisco Bay Area)); 2019-2023 history from the UltraTraffic_CL archive, later weeks from the PeMS clearinghouse station_5min files"
source_url: "https://pems.dot.ca.gov"
citation: "n/a (no dataset paper; TSFLab real-time track)"
citation_url: "n/a"
license: "unknown"
redistribution: "unknown"
frequency: "hourly (1h)"
time_span: "2019-01-01 00:00 to 2023-12-31 23:00 in release 2026.09.28-2003 (grows with each weekly release)"
length: 43824
channels: 2472
channel_kind: "nodes"
target: "total flow of every station"
missing_values: "2.15% of cells are NaN in the stored panel (measured); the loader forward-fills them, back-filling only a series' start"
protocol: "TSFLab static protocol on a frozen release: chronological 7:1:2 split, one z-score from the training rows, seq_len 168 and horizon 24 as in the real-time track; the live rolling rounds are the separate real-time evaluation"
seq_lens: [168]
pred_lens: [24]
split: "7:1:2"
stats_basis: "measured"
related: ["ultratraffic_ba_st", "ultratraffic_ba_ts", "rt/traffic_pems_la", "rt/traffic_pems_sac", "rt/traffic_pems_sb"]
realtime_track: "traffic_pems_ba"
config: "configs/datasets/rt/traffic_pems_ba.toml"
loader: "realtime_panel_st"
alias: "rt_traffic_pems_ba"
task_modes: ["spatiotemporal", "covariate"]
---

# rt/traffic_pems_ba

## Overview

`rt_traffic_pems_ba` serves the hourly total vehicle flow of every station in Caltrans PeMS District 4 (San Francisco Bay Area) as a static dataset: the real-time panel store `dataset/realtime/traffic_pems_ba` is read at release `2026.09.28-2003` and exposed in the spatiotemporal layout (stations as nodes, plus time-of-day and day-of-week covariates). It is the same panel a real-time round uses as history, so a model trained here can be compared with the live `traffic_pems_ba` track.

## Provenance and license

- Source system: Caltrans PeMS (https://pems.dot.ca.gov), District 4. Its Conditions of Use say site information is in the public domain unless otherwise indicated; this is a generic policy, not a statement about the packaged data.
- The 2019-2023 bootstrap history comes from the UltraTraffic_CL archive (publisher not identified in the repository; see `ultratraffic_ba_st`); weekly increments come from the PeMS clearinghouse `station_5min` files summed to hourly flow (free PeMS account).
- `license` and `redistribution` are `unknown`; confirm the archive's terms before publishing a release or this preset's data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Release | `2026.09.28-2003` (bootstrap release; the manifest holds this one release) | measured |
| Rows | 43,824 hourly (2019-01-01 00:00 to 2023-12-31 23:00; a contiguous hourly grid) | measured |
| Stations | 2,472 | measured |
| NaN cells in the store | 2.15% (per year 2019-2023: 4.3, 3.0, 2.5, 0.9, 0.0 %) | measured |
| Stations with more than 50% NaN | 62 | measured |
| Stations whose first reading is after the training split ends | 23 | measured |
| Value range | 0 to 15,101; mean 2710.0, standard deviation 1914.7 (vehicles per hour, NaN excluded) | measured |
| Exact zeros | 0.57% of observed values | measured |
| Mean flow by year | 2,871 (2019), 2,438 (2020), 2,714 (2021), 2,747 (2022), 2,780 (2023) | measured |
| Track config | `configs/realtime/traffic_pems_ba.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.8, timezone America/Los_Angeles | source-reported (config) |

Measured from the local store `dataset/realtime/traffic_pems_ba` (read-only). With 7:1:2 the training rows end 2022-07-02 and the validation rows end 2022-12-31; the test split is therefore all of 2023, the complete-coverage year.

## Standard protocol and known pitfalls

- **Coverage is not uniform.** The NaN share falls from 4.3% in 2019 to 0.0% in 2023, so the test split is the best-covered year while the training split carries most of the filled gaps.
- **2020 regime.** Mean flow dips in 2020 and does not return to its 2019 level in every district (see the yearly means above), so the training split mixes pre-pandemic and pandemic regimes.
- **No adjacency.** The store carries no station coordinates, so graph models receive no adjacency.
- **Frozen snapshot, not the live track.** The preset reads the local panel store at the pinned release (`version`); the live track keeps growing and its rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation. A static split of the snapshot says nothing about data that arrive after it.
- **Gap filling.** Unobserved cells are forward-filled and only a series' leading gap is back-filled from its first reading; the loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training. Fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `realtime_panel_st`
- Config: [`configs/datasets/rt/traffic_pems_ba.toml`](../../../../configs/datasets/rt/traffic_pems_ba.toml)
- Local path: `./dataset/realtime/traffic_pems_ba`
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
  "track": "traffic_pems_ba",
  "version": "2026.09.28-2003"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf data inspect --config configs/datasets/rt/traffic_pems_ba.toml`. Not published as files. The panel store is written by the real-time track (see `docs/en/realtime.md` and `tsf realtime --help`) under the local path above, or pulled from a Hub revision via `dataset.params.revision`. Reference this preset from an experiment configuration rather than duplicating its loader parameters.

## Composition constraints

Task modes: `spatiotemporal`, `covariate`. Match one of them to the model's declared task mode; the
config loader rejects other combinations. Change feature or scaling parameters
only after reading the loader. Paths are repository defaults and may need local
overrides; the card does not imply that the data is bundled.
<!-- dataset-card:canonical:end -->

## Related datasets

- [`ultratraffic_ba_st`](../../ultratraffic_ba_st/README.md): the same district, 2023 only, loaded from the UltraTraffic store
- [`ultratraffic_ba_ts`](../../ultratraffic_ba_ts/README.md): the same data with stations as plain channels
- [`rt/traffic_pems_la`](../traffic_pems_la/README.md): another PeMS district as a real-time preset
- [`rt/traffic_pems_sac`](../traffic_pems_sac/README.md): another PeMS district as a real-time preset
- [`rt/traffic_pems_sb`](../traffic_pems_sb/README.md): another PeMS district as a real-time preset

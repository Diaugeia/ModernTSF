---
name: "rt/grid_ercot"
kind: "dataset"
summary: "Hourly electricity load of the eight ERCOT weather zones in Texas from 2019; frozen release of the real-time track grid_ercot, seq_len 168 and horizon 24 in America/Chicago local time."
domain: "Energy / electricity load"
tags: ["electricity", "load", "ercot", "texas", "weather-zones", "hourly", "real-time", "spatiotemporal"]
source: "ERCOT hourly load by weather zone: Native_Load archives, then MIS report NP6-345-CD"
source_url: "https://www.ercot.com/gridinfo/load/load_hist"
citation: "n/a (no dataset paper; TSFLab real-time track)"
citation_url: "n/a"
license: "unknown"
redistribution: "unknown"
frequency: "hourly (1h)"
time_span: "2019-01-01 onward (grows with each weekly release)"
length: "grows weekly; bootstrap history starts 2019-01-01"
channels: "8 (COAST, EAST, FWEST, NORTH, NCENT, SOUTH, SCENT, WEST)"
channel_kind: "nodes"
target: "hourly load per weather zone (megawatts per ERCOT; unit not re-measured)"
missing_values: "none expected for the archive; the repeated hour of the autumn clock change is averaged"
protocol: "TSFLab static protocol on a frozen release: chronological 7:1:2 split, one z-score from the training rows, seq_len 168 and horizon 24 as in the real-time track; the live rolling rounds are the separate real-time evaluation"
seq_lens: [168]
pred_lens: [24]
split: "7:1:2"
stats_basis: "source-reported"
related: ["rt/grid_eia_us", "electricity"]
realtime_track: "grid_ercot"
config: "configs/datasets/rt/grid_ercot.toml"
loader: "realtime_panel_st"
alias: "rt_grid_ercot"
task_modes: ["spatiotemporal", "covariate"]
---

# rt/grid_ercot

## Overview

`rt_grid_ercot` serves the eight ERCOT weather-zone load series as a static dataset in the spatiotemporal layout (zones as nodes plus calendar covariates), read from the panel store at a fixed release.

## Provenance and license

- Source: ERCOT public load data (`Native_Load` archives, refreshed monthly, and the MIS report NP6-345-CD). `docs/en/realtime.md` calls it public data; no license text was verified, so `license` and `redistribution` are `unknown`.
- Releases are mirrored to the Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`grid_ercot/`).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Channels | 8 weather zones | source-reported (config) |
| Bootstrap | `Native_Load` archives from 2019-01-01 | source-reported (config) |
| Track config | `configs/realtime/grid_ercot.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.9, timezone America/Chicago | source-reported (config) |

No local copy of `dataset/realtime/grid_ercot` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

## Standard protocol and known pitfalls

- **Local clock time.** Hour-ending values are stored at the start of the hour they cover in America/Chicago local time; the spring-forward hour is absent and the repeated autumn hour is averaged, so the grid is not strictly uniform in UTC.
- **Only eight channels.** The panel is small for spatiotemporal models; it behaves like a multivariate load series.

- **Frozen snapshot, not the live track.** The preset reads the local panel store at a fixed release (`version`); the live track keeps growing and its rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation. A static split of the snapshot says nothing about data that arrive after it.
- **Gap filling.** Unobserved cells are forward-filled and only a series' leading gap is back-filled from its first reading; the loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training. Fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `realtime_panel_st`
- Config: [`configs/datasets/rt/grid_ercot.toml`](../../../../configs/datasets/rt/grid_ercot.toml)
- Local path: `./dataset/realtime/grid_ercot`
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
  "track": "grid_ercot"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/rt/grid_ercot.toml`; fetch
published files with `tsf dataset download rt/grid_ercot` when the preset is
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

- [`rt/grid_eia_us`](../grid_eia_us/README.md): US balancing-authority demand
- [`electricity`](../../electricity/README.md): static electricity-consumption set

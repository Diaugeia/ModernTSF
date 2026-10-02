---
name: "pems_bay"
kind: "dataset"
summary: "PEMS-BAY: 5-minute traffic speed from 325 Caltrans PeMS sensors in the San Francisco Bay Area (first half of 2017), packaged by the DCRNN authors."
domain: "Transportation / road traffic"
tags: ["traffic", "graph", "spatiotemporal", "5min", "california", "sensors", "adjacency", "benchmark", "nodes", "pems_bay", "speed", "bay-area", "dcrnn", "pems-bay"]
source: "Caltrans PeMS; packaged by Li, Yu, Shahabi and Liu (DCRNN)"
source_url: "https://github.com/liyaguang/DCRNN"
citation: "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting (Li et al., ICLR 2018)"
citation_url: "https://arxiv.org/abs/1707.01926"
license: "unknown"
redistribution: "unknown"
frequency: "5-minute (5min)"
time_span: "2017-01-01 to about 2017-06-30 (52,116 steps; the DCRNN paper text says to 2017-05-31)"
length: 52116
channels: 325
channel_kind: "nodes"
target: "traffic speed"
missing_values: "not verified"
protocol: "TSFLab: chronological 7:1:2 split, scaling fitted on the training split only; lookback 12, horizons 12"
literature_protocol: "DCRNN: 12 steps in, 12 steps out (reported at 15/30/60 minutes), chronological 7:1:2, masked MAE/RMSE/MAPE"
seq_lens: [12]
pred_lens: [12]
split: "7:1:2"
stats_basis: "source-reported"
related: ["metr_la", "pems08", "ultratraffic_ba_st"]
config: "configs/datasets/pems_bay.toml"
loader: "cauair_st"
alias: "pems_bay"
task_modes: ["spatiotemporal", "covariate"]
---

# pems_bay

## Overview

PEMS-BAY is a spatiotemporal traffic benchmark: 325 sensors recording traffic speed every five minutes over six months of 2017 (52,116 steps), with a sensor graph. The data come from Caltrans PeMS. In this repository it loads through the `cauair_st` node loader as a `(T, N, 3)` bundle (value, time-in-day, day-in-week) with an adjacency matrix, so graph and spatiotemporal models can use the road network.

## Provenance and license

- Producer: DCRNN authors packaged Caltrans PeMS data; repository https://github.com/liyaguang/DCRNN (code MIT). The packaged file has no stated data license; PeMS's own Conditions of Use say site information is in the public domain unless otherwise indicated, a generic policy, so `license` stays `unknown`.
- Cite: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting (Li et al., ICLR 2018), https://arxiv.org/abs/1707.01926.
- Obtain `pems-bay.h5` and `adj_mx_bay.pkl` from the DCRNN README; TSFLab does not ship them.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Sensors | 325 | source-reported (DCRNN) |
| Steps | 52,116 at 5 minutes | source-reported (TFB Table 5; file not re-counted) |
| Span | paper: 2017-01-01 to 2017-05-31; step count implies to about 2017-06-30 | source-reported (conflicting) |
| Quantity | traffic speed | source-reported |

The preset reads a converted node bundle (`his.npz` with `data` shaped `(T, N, 3)`, `adj_mx.npy`, and `idx_train/val/test.npy`) produced by `tsf dataset convert-traffic`; the repository neither ships nor pins it, so the numbers above are those of the public distribution, and the converted bundle must be inspected before use.

## Standard protocol and known pitfalls

- **Protocol.** DCRNN predicts 12 steps from 12 steps with a 70/10/20 split; the `[12]` fields above are that protocol, not a preset default. The DCRNN paper's date range conflicts with its step count (52,116 steps is about 181 days, i.e. January to June).
- **Bundle contract.** `input_dim = 3` keeps the value plus two calendar covariates (time-in-day, day-in-week) appended by `convert-traffic --add-time`; `scale = false` leaves values unscaled, so scale the value channel upstream or flip `scale` if the model expects z-scored input.
- **Window split.** Window centres are split chronologically by `convert-traffic --splits` (default 0.7,0.1,0.2). The converter also stores whole-series `mean`/`std` in `his.npz`, so any `scale = true` variant would use statistics that include validation and test data.
- **Adjacency.** `adj_mx.npy` comes from the converter's `--adj` input; check how it was built before comparing graph models across papers.
- **Metrics.** The repository evaluator has no masked-metric option; DCRNN-style papers mask zero (missing) targets, so unmasked numbers are not comparable.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `cauair_st`
- Config: [`configs/datasets/pems_bay.toml`](../../../configs/datasets/pems_bay.toml)
- Local path: `./dataset/pems_bay`
- Dataset id: `(not applicable)`
- Track: `standard`

## Input and output contract

Each item is `(value_history, value_future, covariate_history, covariate_future)`; values use `[time, nodes]` and covariates `[time, nodes, features]` before batching.

Sequence length, label length, feature mode, and batch size are supplied by the
experiment task unless explicitly overridden below.

## Dataset parameters

```json
{
  "input_dim": 3,
  "npz_name": "his.npz",
  "scale": false
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/pems_bay.toml`; fetch
published files with `tsf dataset download pems_bay` when the preset is
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

- [`metr_la`](../metr_la/README.md): other DCRNN speed benchmark
- [`pems08`](../pems08/README.md): flow benchmark with the same loader
- [`ultratraffic_ba_st`](../ultratraffic_ba_st/README.md): hourly flow for the same Bay Area district (District 4), 2023

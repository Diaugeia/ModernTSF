---
name: "metr_la"
kind: "dataset"
summary: "METR-LA: 5-minute traffic speed from 207 loop detectors on Los Angeles County highways (March to June 2012), the standard speed benchmark from DCRNN."
domain: "Transport / road traffic"
tags: ["traffic", "graph", "spatiotemporal", "5min", "california", "sensors", "adjacency", "benchmark", "nodes", "metr_la", "speed", "los-angeles", "dcrnn", "metr-la"]
source: "Los Angeles County loop detectors (via Jagadish et al., 2014); packaged by Li, Yu, Shahabi and Liu (DCRNN)"
source_url: "https://github.com/liyaguang/DCRNN"
citation: "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting (Li et al., ICLR 2018)"
citation_url: "https://arxiv.org/abs/1707.01926"
license: "unknown"
redistribution: "unknown"
frequency: "5-minute (5min)"
time_span: "2012-03-01 to 2012-06-27 (file); the DCRNN paper says to 2012-06-30"
length: 34272
channels: 207
channel_kind: "nodes"
target: "traffic speed"
missing_values: "zeros mark missing readings in the DCRNN file (practice, not stated in its README); the TFB copy has no NaN"
protocol: "TSFLab: chronological 7:1:2 split, scaling fitted on the training split only; lookback 12, horizons 12"
literature_protocol: "DCRNN: 12 steps in, 12 steps out (reported at 15/30/60 minutes), chronological 7:1:2, masked MAE/RMSE/MAPE"
seq_lens: [12]
pred_lens: [12]
split: "7:1:2"
stats_basis: "source-reported"
related: ["pems_bay", "pems04", "traffic"]
config: "configs/datasets/metr_la.toml"
loader: "cauair_st"
alias: "metr_la"
task_modes: ["spatiotemporal", "covariate"]
---

# metr_la

## Overview

METR-LA is a spatiotemporal traffic benchmark: 207 sensors recording traffic speed every five minutes over about four months in 2012 (34,272 steps), with a sensor graph. Zeros are commonly treated as missing readings (community practice, not verified here). In this repository it loads through the `cauair_st` node loader as a `(T, N, 3)` bundle (value, time-in-day, day-in-week) with an adjacency matrix, so graph and spatiotemporal models can use the road network.

## Provenance and license

- Producer: DCRNN authors (USC and Caltech) packaged LA County loop-detector data; repository https://github.com/liyaguang/DCRNN (code MIT). The data file (`metr-la.h5` on Google Drive) has no stated data license, so `license` is `unknown`.
- Cite: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting (Li et al., ICLR 2018), https://arxiv.org/abs/1707.01926.
- Obtain `metr-la.h5` and `adj_mx.pkl` from the DCRNN README; TSFLab does not ship them (see `tsf dataset convert-traffic --help`).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Sensors | 207 | source-reported (DCRNN) |
| Steps | 34,272 at 5 minutes (= 119 days) | source-reported (TFB data file) |
| Span | 2012-03-01 to 2012-06-27 23:55 (paper: to 06-30; conflicting) | source-reported |
| Quantity | traffic speed | source-reported |
| Missing values | zeros mark missing in the original h5; none as NaN in the TFB copy | source-reported |

The preset reads a converted node bundle (`his.npz` with `data` shaped `(T, N, 3)`, `adj_mx.npy`, and `idx_train/val/test.npy`) produced by `tsf dataset convert-traffic`; the repository neither ships nor pins it, so the numbers above are those of the public distribution, and the converted bundle must be inspected before use.

## Standard protocol and known pitfalls

- **Protocol.** DCRNN predicts 12 steps from 12 steps (horizons 3, 6, 12 reported) with a 70/10/20 split; the `[12]` fields above are that protocol, not a preset default.
- **Bundle contract.** `input_dim = 3` keeps the value plus two calendar covariates (time-in-day, day-in-week) appended by `convert-traffic --add-time`; `scale = false` leaves values unscaled, so scale the value channel upstream or flip `scale` if the model expects z-scored input.
- **Window split.** Window centres are split chronologically by `convert-traffic --splits` (default 0.7,0.1,0.2). The converter also stores whole-series `mean`/`std` in `his.npz`, so any `scale = true` variant would use statistics that include validation and test data.
- **Adjacency.** `adj_mx.npy` comes from the converter's `--adj` input; check how it was built before comparing graph models across papers.
- **Metrics.** The repository evaluator has no masked-metric option; DCRNN-style papers mask zero (missing) targets, so unmasked numbers are not comparable.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `cauair_st`
- Config: [`configs/datasets/metr_la.toml`](../../../configs/datasets/metr_la.toml)
- Local path: `./dataset/metr_la`
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

Inspect availability with `tsf dataset inspect --config configs/datasets/metr_la.toml`; fetch
published files with `tsf dataset download metr_la` when the preset is
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

- [`pems_bay`](../pems_bay/README.md): other DCRNN speed benchmark
- [`pems04`](../pems04/README.md): flow benchmark with the same loader
- [`traffic`](../traffic/README.md): hourly LTSF traffic set without a graph

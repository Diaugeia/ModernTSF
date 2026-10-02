---
name: "pems04"
kind: "dataset"
summary: "PEMS04: 5-minute traffic flow from 307 Caltrans PeMS District 4 sensors (2018-01-01 to 2018-02-28), the STSGCN/ASTGCN benchmark for spatiotemporal graph forecasting."
domain: "Transportation / road traffic"
tags: ["traffic", "graph", "spatiotemporal", "5min", "california", "sensors", "adjacency", "benchmark", "nodes", "pems04", "flow", "district-4", "stsgcn", "astgcn"]
source: "Caltrans PeMS District 4; processed by Song et al. (STSGCN) and Guo et al. (ASTGCN)"
source_url: "https://github.com/Davidham3/STSGCN"
citation: "Spatial-Temporal Synchronous Graph Convolutional Networks (Song et al., AAAI 2020)"
citation_url: "https://ojs.aaai.org/index.php/AAAI/article/view/5438"
license: "unknown"
redistribution: "unknown"
frequency: "5-minute (5min)"
time_span: "2018-01-01 to 2018-02-28"
length: 16992
channels: 307
channel_kind: "nodes"
target: "traffic flow"
missing_values: "not stated in the papers"
protocol: "STSGCN: 12 steps in, 12 steps out (one hour), chronological 6:2:2, MAE/RMSE/MAPE"
seq_lens: [12]
pred_lens: [12]
split: "6:2:2"
stats_basis: "source-reported"
related: ["pems03", "pems07", "metr_la"]
config: "configs/datasets/pems04.toml"
loader: "cauair_st"
alias: "pems04"
task_modes: ["spatiotemporal", "covariate"]
---

# pems04

## Overview

PEMS04 is a spatiotemporal traffic benchmark: 307 sensors recording traffic flow every five minutes over 2018-01-01 to 2018-02-28, with a sensor graph. It covers PeMS District 4 (San Francisco Bay Area). In this repository it loads through the `cauair_st` node loader as a `(T, N, 3)` bundle (value, time-in-day, day-in-week) with an adjacency matrix, so graph and spatiotemporal models can use the road network.

## Provenance and license

- Producer: Caltrans PeMS District 4 data, processed by Song et al. (STSGCN, https://github.com/Davidham3/STSGCN) and Guo et al. (ASTGCN, https://github.com/Davidham3/ASTGCN, AAAI 2019). Neither repository carries a license, so the packaged data's license is `unknown`; PeMS's Conditions of Use call site information public domain unless otherwise indicated (generic policy).
- Cite Spatial-Temporal Synchronous Graph Convolutional Networks (Song et al., AAAI 2020), https://ojs.aaai.org/index.php/AAAI/article/view/5438, and the ASTGCN paper (https://ojs.aaai.org/index.php/AAAI/article/view/3881).
- TSFLab does not ship the data (the STSGCN files are distributed through Baidu Pan); convert with `tsf dataset convert-traffic`.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Sensors | 307 | source-reported (STSGCN Table 1) |
| Span | 2018-01-01 to 2018-02-28 | source-reported (STSGCN) |
| Steps | 16,992 | source-reported (TFB data file) |
| Frequency | 5 minutes (12 per hour) | source-reported |
| Quantity | traffic flow (the ASTGCN files for PEMS04/08 also carry occupancy and speed; STSGCN and this preset use flow only) | source-reported |

The preset reads a converted node bundle (`his.npz` with `data` shaped `(T, N, 3)`, `adj_mx.npy`, and `idx_train/val/test.npy`) produced by `tsf dataset convert-traffic`; the repository neither ships nor pins it, so the numbers above are those of the public distribution, and the converted bundle must be inspected before use.

## Standard protocol and known pitfalls

- **Split mismatch.** STSGCN uses 6:2:2, while `tsf dataset convert-traffic` defaults to 7:1:2; pass `--splits 0.6,0.2,0.2` to match the literature. The `[12]` horizon fields above are the STSGCN protocol, not a preset default.
- **Bundle contract.** `input_dim = 3` keeps the value plus two calendar covariates (time-in-day, day-in-week) appended by `convert-traffic --add-time`; `scale = false` leaves values unscaled, so scale the value channel upstream or flip `scale` if the model expects z-scored input.
- **Window split.** Window centres are split chronologically by `convert-traffic --splits` (default 0.7,0.1,0.2). The converter also stores whole-series `mean`/`std` in `his.npz`, so any `scale = true` variant would use statistics that include validation and test data.
- **Adjacency.** `adj_mx.npy` comes from the converter's `--adj` input; check how it was built before comparing graph models across papers.
- **Metrics.** The repository evaluator has no masked-metric option; DCRNN-style papers mask zero (missing) targets, so unmasked numbers are not comparable.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `cauair_st`
- Config: [`configs/datasets/pems04.toml`](../../../configs/datasets/pems04.toml)
- Local path: `./dataset/pems04`
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

Inspect availability with `tsf dataset inspect --config configs/datasets/pems04.toml`; fetch
published files with `tsf dataset download pems04` when the preset is
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

- [`pems03`](../pems03/README.md): sibling STSGCN flow benchmark
- [`pems07`](../pems07/README.md): sibling STSGCN flow benchmark
- [`metr_la`](../metr_la/README.md): DCRNN speed benchmark with the same loader

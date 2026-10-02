---
name: "pems07"
kind: "dataset"
summary: "PEMS07: 5-minute traffic flow from 883 Caltrans PeMS District 7 sensors (2017-05-01 to 2017-08-31), the STSGCN/ASTGCN benchmark for spatiotemporal graph forecasting."
domain: "Transport / road traffic"
tags: ["traffic", "graph", "spatiotemporal", "5min", "california", "sensors", "adjacency", "benchmark", "nodes", "pems07", "flow", "district-7", "stsgcn", "astgcn"]
source: "Caltrans PeMS District 7; processed by Song et al. (STSGCN) and Guo et al. (ASTGCN)"
source_url: "https://github.com/Davidham3/STSGCN"
citation: "Spatial-Temporal Synchronous Graph Convolutional Networks (Song et al., AAAI 2020)"
citation_url: "https://ojs.aaai.org/index.php/AAAI/article/view/5438"
license: "Public domain unless otherwise indicated (Caltrans PeMS Conditions of Use; raw source only, the packaged copy states no license)"
redistribution: "conditional"
frequency: "5-minute (5min)"
time_span: "2017-05-01 to 2017-08-31"
length: "not verified (28,224 is commonly cited; the stated dates imply 35,424)"
channels: 883
channel_kind: "nodes"
target: "traffic flow"
missing_values: "not stated in the papers"
protocol: "TSFLab: chronological 7:1:2 split, values unscaled by default (scale = false, calendar covariates kept raw); lookback 12, horizons 12"
literature_protocol: "STSGCN: 12 steps in, 12 steps out (one hour), chronological 6:2:2, MAE/RMSE/MAPE"
seq_lens: [12]
pred_lens: [12]
split: "7:1:2"
stats_basis: "source-reported"
related: ["pems03", "pems04", "metr_la"]
config: "configs/datasets/pems07.toml"
loader: "cauair_st"
alias: "pems07"
task_modes: ["spatiotemporal", "covariate"]
---

# pems07

## Overview

PEMS07 is a spatiotemporal traffic benchmark: 883 sensors recording traffic flow every five minutes over 2017-05-01 to 2017-08-31, with a sensor graph. It covers PeMS District 7 (Los Angeles area). In this repository it loads through the `cauair_st` node loader as a `(T, N, 3)` bundle (value, time-in-day, day-in-week) with an adjacency matrix, so graph and spatiotemporal models can use the road network.

## Provenance and license

- Producer: Caltrans PeMS District 7 data, processed by Song et al. (STSGCN, https://github.com/Davidham3/STSGCN) and Guo et al. (ASTGCN, https://github.com/Davidham3/ASTGCN, AAAI 2019). Neither repository carries a license (GitHub API reports none for STSGCN; ASTGCN redirects to a repository without one), so the packaged files have no stated license. Caltrans PeMS Conditions of Use (https://pems.dot.ca.gov/?view=tou) say: "In general, information presented on this web site, unless otherwise indicated, is considered in the public domain", and that to use information "not owned or created by the State, you must seek permission directly from the owning (or holding) sources". That covers the raw PeMS data; the packaged copy below carries no license of its own, so redistribute it only after confirming the packager's terms.
- Cite Spatial-Temporal Synchronous Graph Convolutional Networks (Song et al., AAAI 2020), https://ojs.aaai.org/index.php/AAAI/article/view/5438, and the ASTGCN paper (https://ojs.aaai.org/index.php/AAAI/article/view/3881).
- TSFLab does not ship the data (the STSGCN files are distributed through Baidu Pan); convert with `tsf data prepare --from traffic`.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Sensors | 883 | source-reported (STSGCN Table 1) |
| Span | 2017-05-01 to 2017-08-31 | source-reported (STSGCN) |
| Steps | not verified (28,224 is commonly cited; the stated dates imply 35,424) | unresolved |
| Frequency | 5 minutes (12 per hour) | source-reported |
| Quantity | traffic flow (the ASTGCN files for PEMS04/08 also carry occupancy and speed; STSGCN and this preset use flow only) | source-reported |

The preset reads a converted node bundle (`his.npz` with `data` shaped `(T, N, 3)`, `adj_mx.npy`, and `idx_train/val/test.npy`) produced by `tsf data prepare --from traffic`; the repository neither ships nor pins it, so the numbers above are those of the public distribution, and the converted bundle must be inspected before use.

## Standard protocol and known pitfalls

- **Split.** TSFLab uses a chronological 7:1:2 split for this dataset. Results reported under other splits (6:2:2) are not directly comparable.
- **Bundle contract.** `input_dim = 3` keeps the value plus two calendar covariates (time-in-day, day-in-week) appended by `tsf data prepare --from traffic --add-time`; `scale = false` leaves values unscaled, so scale the value channel upstream or flip `scale` if the model expects z-scored input.
- **Window split.** Window centres are split chronologically by `tsf data prepare --from traffic --splits` (default 0.7,0.1,0.2); the window indices depend on the converter's `--seq-len` and `--pred-len` (default 96 each), so convert with 12 and 12 for the 12-in/12-out protocol and rebuild the bundle to change either. The converter also stores whole-series `mean`/`std` in `his.npz`, so any `scale = true` variant would use statistics that include validation and test data.
- **Adjacency.** `adj_mx.npy` comes from the converter's `--adj` input; check how it was built before comparing graph models across papers.
- **Metrics.** The repository evaluator has no masked-metric option; DCRNN-style papers mask zero (missing) targets, so unmasked numbers are not comparable.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `cauair_st`
- Config: [`configs/datasets/pems07.toml`](../../../configs/datasets/pems07.toml)
- Local path: `./dataset/pems07`
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

Inspect availability with `tsf data inspect --config configs/datasets/pems07.toml`; fetch
published files with `tsf data download pems07` when the preset is
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

- [`pems03`](../pems03/README.md): sibling STSGCN flow benchmark
- [`pems04`](../pems04/README.md): sibling STSGCN flow benchmark
- [`metr_la`](../metr_la/README.md): DCRNN speed benchmark with the same loader

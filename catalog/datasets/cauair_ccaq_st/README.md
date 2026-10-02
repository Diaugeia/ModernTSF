---
name: "cauair_ccaq_st"
kind: "dataset"
summary: "CCAQ nationwide city air-quality forecasting data (209 sites, hourly, 28 months) in the CauAir repository layout, with per-station covariates; the local copy is a synthetic fixture."
domain: "Environment / air quality"
tags: ["air-quality", "aqi", "pm2.5", "china", "ccaq", "cauair", "gagnn", "spatiotemporal", "covariate", "hourly", "stations", "synthetic-fixture", "graph"]
source: "Chen et al. (GAGNN, ACM TKDD 2023) CCAQ data as processed in the CauAir repository (Ma et al.)"
source_url: "https://github.com/PoorOtterBob/CauAir"
citation: "Causal Learning Meet Covariates: Empowering Lightweight and Effective Nationwide Air Quality Forecasting (Ma et al., IJCAI 2025)"
citation_url: "https://www.ijcai.org/proceedings/2025/353"
license: "unknown"
redistribution: "unknown"
frequency: "hourly (1h)"
time_span: "28 months (source-reported; start date not recorded here)"
length: 20373
channels: 209
channel_kind: "stations"
target: "air quality index value (channel 0 of `data`)"
missing_values: "unknown"
protocol: "CauAir: 24 steps in, 24 steps out, index-windowed train/val/test split shipped with the data (ratios not stated); weather and calendar covariates"
seq_lens: [24]
pred_lens: [24]
split: "index files shipped with the data (ratios not stated)"
stats_basis: "mixed"
related: ["cauair_ccaq_ts", "smoke", "metr_la"]
config: "configs/datasets/cauair_ccaq_st.toml"
loader: "cauair_st"
alias: "cauair_ccaq_st"
task_modes: ["spatiotemporal", "covariate"]
---

# cauair_ccaq_st

## Overview

CCAQ is a nationwide city air-quality dataset of hourly station readings with weather covariates. CauAir (IJCAI 2025) did not introduce it: its Table 1 credits Chen et al. (GAGNN, ACM TKDD 2023), and the CauAir repository ships it as processed `his.npz` plus index files. CauAir's own dataset is LargeAQ (1,341 stations, 2016-2023, hourly), which this repository has no preset for. This preset reads the bundle from `dataset/cauair_ccaq` with the `cauair_st` loader: Station values `[time, stations]` plus per-station covariates `[time, stations, features]` are kept.

**Local data caveat.** The `dataset/cauair_ccaq` directory in a development checkout is a synthetic fixture written by `scripts/make_smoke_data.py` (400 steps, 8 nodes, 8 channels, ring adjacency), not real CCAQ data.

## Provenance and license

- Dataset origin: Chen et al., Group-aware graph neural network for nationwide city air quality forecasting, ACM TKDD 2023 (GAGNN); CauAir (https://github.com/PoorOtterBob/CauAir) ships the processed data. Raw air-quality readings come from China's national monitoring network.
- Cite Causal Learning Meet Covariates: Empowering Lightweight and Effective Nationwide Air Quality Forecasting (Ma et al., IJCAI 2025) (https://www.ijcai.org/proceedings/2025/353) and the GAGNN paper.
- License: the CauAir repository has no LICENSE file (GitHub API 404) and the README states none; the upstream monitoring-data terms were not checked. `license` and `redistribution` are `unknown`.
- The CauAir model card records the same repository as reference-only; no data are bundled with TSFLab.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Sites (real CCAQ) | 209 | source-reported (CauAir Table 1) |
| Steps (real CCAQ) | 20,373 hourly (28 months) | source-reported |
| Local fixture `data` | 400 steps x 8 nodes x 8 channels (value, time-in-day, day-in-week, five covariates) | measured |
| Local fixture splits | window centres: train 245, validation 35, test 71 | measured |
| Local adjacency | 8 x 8 ring (each node linked to two neighbours and itself) | measured |

Measured on `dataset/cauair_ccaq` (read-only); real-CCAQ rows are source-reported and not verifiable from the local fixture.

## Standard protocol and known pitfalls

- **Fixture versus benchmark.** Metrics on the local fixture are meaningless; real CCAQ numbers need the real bundle placed in `dataset/cauair_ccaq`.
- **Bundle contract.** Channel 0 of `data` is the target; channels 1 onward are per-node covariates. A window centred at `i` covers history `[i - seq_len + 1, i]` and future `[i + 1, i + horizon]`; splits come from `idx_train/val/test.npy` and cannot be changed through `split_ratio`.
- **Scaling.** `scale = true` z-scores with the `mean`/`std` stored in the bundle (taken as given; how they were computed for real CCAQ is not recorded). `input_dim = 8` keeps the first eight channels.
- **Covariates.** Future covariates are part of the input contract (covariate task mode); a model that ignores them is not using the information CauAir was built around.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `cauair_st`
- Config: [`configs/datasets/cauair_ccaq_st.toml`](../../../configs/datasets/cauair_ccaq_st.toml)
- Local path: `./dataset/cauair_ccaq`
- Dataset id: `(not applicable)`
- Track: `standard`

## Input and output contract

Each item is `(value_history, value_future, covariate_history, covariate_future)`; values use `[time, nodes]` and covariates `[time, nodes, features]` before batching.

Sequence length, label length, feature mode, and batch size are supplied by the
experiment task unless explicitly overridden below.

## Dataset parameters

```json
{
  "input_dim": 8,
  "npz_name": "his.npz",
  "scale": true
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/cauair_ccaq_st.toml`; fetch
published files with `tsf dataset download cauair_ccaq_st` when the preset is
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

- [`cauair_ccaq_ts`](../cauair_ccaq_ts/README.md): same bundle, other layout
- [`smoke`](../smoke/README.md): synthetic fixture written by the same script
- [`metr_la`](../metr_la/README.md): other node-bundle benchmark

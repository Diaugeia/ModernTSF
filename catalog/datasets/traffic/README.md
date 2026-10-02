---
name: "traffic"
kind: "dataset"
summary: "Hourly road occupancy rate (0 to 1) from 862 freeway sensors in the San Francisco Bay Area for 2015-2016, from Caltrans PeMS via the LSTNet/Autoformer packaging."
domain: "Transport / road traffic"
tags: ["traffic", "pems", "road", "occupancy", "california", "bay-area", "ltsf", "benchmark", "multivariate", "hourly", "lstnet", "high-dimensional"]
source: "California Department of Transportation (Caltrans PeMS); packaged by Lai et al. (LSTNet)"
source_url: "http://pems.dot.ca.gov"
citation: "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks (Lai et al., SIGIR 2018)"
citation_url: "https://arxiv.org/abs/1703.07015"
license: "Public domain unless otherwise indicated (Caltrans PeMS Conditions of Use; raw source only, the packaged copy states no license)"
redistribution: "conditional"
frequency: "hourly (1h)"
time_span: "2015-01-01 00:00:01 to 2016-12-31 23:00:01"
length: 17544
channels: 862
channel_kind: "channels"
target: "862 (last sensor column; the TSLib copy names it OT)"
missing_values: "none in this file (measured)"
protocol: "TSFLab: chronological 7:1:2 split, scaling fitted on the training split only; lookback 96/336/512, horizons 96/192/336/720"
literature_protocol: "LTSF: lookback 96 (336 and 512 also searched), horizons 96/192/336/720, chronological 7:1:2, MSE/MAE on z-scored data"
seq_lens: [96, 336, 512]
pred_lens: [96, 192, 336, 720]
split: "7:1:2"
stats_basis: "mixed"
related: ["electricity", "pems04", "ultratraffic_ba_ts"]
config: "configs/datasets/traffic.toml"
loader: "traffic"
alias: "traffic"
task_modes: ["time_series"]
---

# traffic

## Overview

Traffic is the hourly road occupancy rate (the fraction of time a sensor is occupied, between 0 and 1) of 862 loop-detector sensors on San Francisco Bay Area freeways during 2015 and 2016. Caltrans collects it in PeMS; Lai et al. (LSTNet) packaged an hourly version that Informer, Autoformer, and Time-Series-Library adopted. It is the standard 800-plus-channel benchmark with sharp daily and weekly rush-hour patterns, and it has no sensor locations or adjacency in this packaging.

## Provenance and license

- Source system: Caltrans Performance Measurement System (PeMS), http://pems.dot.ca.gov. Caltrans PeMS Conditions of Use (https://pems.dot.ca.gov/?view=tou) say: "In general, information presented on this web site, unless otherwise indicated, is considered in the public domain", and that to use information "not owned or created by the State, you must seek permission directly from the owning (or holding) sources". That covers the raw PeMS data; the packaged copy below carries no license of its own, so redistribute it only after confirming the packager's terms.
- Packaging: Lai et al. 2018, https://github.com/laiguokun/multivariate-time-series-data, which has no license file (GitHub API: none) and no data terms in its README; the preprocessed copy therefore carries no license of its own. Raw source and packaging are separate: the PeMS public-domain policy applies to the raw data, but redistribute the LSTNet copy only after weighing both, hence `conditional`.
- Cite LSTNet (SIGIR 2018) and credit Caltrans PeMS.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 17,544 | measured |
| Channels | 862 sensors (columns named 1 to 862; no `OT` column) | measured |
| Time span | 2015-01-01 00:00:01 to 2016-12-31 23:00:01, step 1 hour, no gaps | measured |
| Missing values | 0 | measured |
| Value range | 0.0000 to 0.7240, mean 0.0567, std 0.0540 | measured |
| Exact zeros | 0.90% of all values | measured |
| Quantity | occupancy rate in [0, 1]; the 2015-2016 span is 731 days | source-reported (LSTNet) |

Measured on `dataset/traffic/traffic.csv` (read-only).

## Standard protocol and known pitfalls

- **Split.** TSFLab uses a chronological 7:1:2 split for this dataset. Results reported under other splits (6:2:2) are not directly comparable.
- **Target column.** This copy names the sensors `1` to `862` and has no `OT` column, so the preset targets `862`, the last sensor, which the TSLib copy calls `OT`. `"S"` and `"MS"` forecast sensor `862`.
- **Memory.** 862 channels with a long lookback make attention-style models expensive; many papers cap batch size or channel sampling, which changes results.
- **Different dataset, same name.** The PeMS03/04/07/08 presets are 5-minute flow graphs and the `ultratraffic_*` presets are hourly flow per station for 2023; none are comparable with this occupancy file.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `traffic`
- Config: [`configs/datasets/traffic.toml`](../../../configs/datasets/traffic.toml)
- Local path: `./dataset/traffic/traffic.csv`
- Dataset id: `(not applicable)`
- Track: `standard`

## Input and output contract

Each item provides history/target windows and timestamp marks; after batching, values use `[batch, time, channels]`.

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
  "target": "862"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/traffic.toml`; fetch
published files with `tsf dataset download traffic` when the preset is
listed by `tsf dataset download --list`, otherwise place the data at the local
path above (see `tsf dataset prepare --help`). Reference this preset from an
experiment configuration rather than duplicating its loader parameters.

## Composition constraints

Task modes: `time_series`. Match one of them to the model's declared task mode; the
config loader rejects other combinations. Change feature or scaling parameters
only after reading the loader. Paths are repository defaults and may need local
overrides; the card does not imply that the data is bundled.
<!-- dataset-card:canonical:end -->

## Related datasets

- [`electricity`](../electricity/README.md): other hourly LTSF set with hundreds of channels
- [`pems04`](../pems04/README.md): 5-minute PeMS flow graph with adjacency
- [`ultratraffic_ba_ts`](../ultratraffic_ba_ts/README.md): hourly PeMS flow per station, 2023, stations as channels

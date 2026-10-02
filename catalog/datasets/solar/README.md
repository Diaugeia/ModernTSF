---
name: "solar"
kind: "dataset"
summary: "Power output of 137 simulated photovoltaic plants in Alabama for 2006 at 10-minute resolution (NREL solar integration data, packaged by LSTNet), with exact zeros at night."
domain: "Energy / solar power"
tags: ["solar", "photovoltaic", "pv", "alabama", "nrel", "energy", "ltsf", "benchmark", "multivariate", "10min", "lstnet", "zero-inflated"]
source: "NREL (now NLR) Solar Power Data for Integration Studies; packaged by Lai et al. (LSTNet)"
source_url: "https://www.nlr.gov/grid/solar-power-data"
citation: "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks (Lai et al., SIGIR 2018)"
citation_url: "https://arxiv.org/abs/1703.07015"
license: "unknown"
redistribution: "unknown"
frequency: "10-minute (10min)"
time_span: "2006-01-01 to 2006-12-31 (source-reported; the file has no date column)"
length: 52560
channels: 137
channel_kind: "channels"
target: "column \"0\" (first plant) in S mode; all 137 plants in M mode"
missing_values: "none (measured: no NaN); night values are exact zeros, not missing"
protocol: "TSFLab: chronological 7:1:2 split, scaling fitted on the training split only; lookback 96/336/512, horizons 96/192/336/720"
literature_protocol: "LTSF: lookback 96 (336 and 512 also searched), horizons 96/192/336/720; TFB uses chronological 6:2:2 while this preset uses 7:1:2; LSTNet used 6:2:2 with horizons 3/6/12/24 steps"
seq_lens: [96, 336, 512]
pred_lens: [96, 192, 336, 720]
split: "7:1:2"
stats_basis: "mixed"
related: ["electricity", "gift_eval/solar_10T"]
config: "configs/datasets/solar.toml"
loader: "solar"
alias: "solar"
task_modes: ["time_series"]
---

# solar

## Overview

Solar-Energy contains the simulated power output of 137 photovoltaic plants in Alabama for the year 2006. NREL publishes the underlying 5-minute simulations for solar integration studies; LSTNet packaged a 10-minute version that became a standard benchmark for periodic, zero-inflated multivariate series: output follows the day-night cycle and cloud-driven fluctuations and is exactly zero at night.

## Provenance and license

- Producer: NREL (the site now resolves under nlr.gov), Solar Power Data for Integration Studies, https://www.nlr.gov/grid/solar-power-data. The page states no license; it warns that the data are for specific years and are not representative of typical radiation levels or for site-specific project development.
- Packaging: Lai et al. 2018 (LSTNet), https://github.com/laiguokun/multivariate-time-series-data (no license file). Redistribution terms are therefore `unknown`.
- Contradiction to be aware of: NREL describes simulated 5-minute data, while the LSTNet README calls the series records sampled every 10 minutes.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 52,560 (= 365 days x 144 steps) | measured |
| Channels | 137 plants (headerless text file, no date column) | measured |
| Missing values | 0 | measured |
| Value range | 0.0 to 88.9, mean 6.353, std 10.151 | measured |
| Exact zeros | 55.1% of all values; no negative values | measured |
| Night hours (>= 99% zeros, hour of day from row index) | 0, 1, 2, 3, 4, 19, 20, 21, 22, 23 | measured |
| Zero fraction at hours 5 / 6 / 7 / 18 / 19 | 0.86 / 0.45 / 0.10 / 0.96 / 1.00 | measured |
| Span and origin | calendar year 2006, Alabama, simulated PV output | source-reported (LSTNet, NREL) |

Measured on `dataset/solar/solar.txt` (read-only); hour of day is derived assuming row 0 is 00:00.

## Standard protocol and known pitfalls

- **Zeros at night.** About 55% of all values are exactly zero (hours 0-4 and 19-23 are all zero, measured). Relative-error metrics such as MAPE are undefined there, z-scoring turns the night level into a large negative constant, and a large share of the error budget is the day-night mask rather than cloud-driven variation.
- **Split.** TSFLab uses a chronological 7:1:2 split for this dataset. Results reported under other splits (6:2:2) are not directly comparable.
- **No timestamps.** The file has no date column; the loader assigns the same constant calendar mark (all-zero date) to every step, so models that rely on time marks receive no calendar information. Daily periodicity must be learned from the values.
- **Target.** `target = "0"` selects the first plant for `features = "S"`; `"MS"` forecasts the last plant.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `solar`
- Config: [`configs/datasets/solar.toml`](../../../configs/datasets/solar.toml)
- Local path: `./dataset/solar/solar.txt`
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
  "target": "0"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/solar.toml`; fetch
published files with `tsf dataset download solar` when the preset is
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

- [`electricity`](../electricity/README.md): other periodic energy set, 7:1:2
- [`gift_eval/solar_10T`](../gift_eval/solar_10T/README.md): GIFT-Eval packaging of the same plants, different protocol

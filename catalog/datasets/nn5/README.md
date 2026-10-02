---
name: "nn5"
kind: "dataset"
summary: "Daily cash withdrawals from 111 ATMs in the NN5 forecasting competition, in the Monash/TFB multivariate packaging; a short, weekly-seasonal benchmark."
domain: "Finance / retail banking"
tags: ["nn5", "atm", "cash-withdrawal", "banking", "daily", "competition", "monash", "tfb", "multivariate", "weekly-seasonality", "short-series"]
source: "NN5 neural-network forecasting competition (Crone et al.); Monash archive (Godahewa et al.) and TFB packaging"
source_url: "https://zenodo.org/records/3889740"
citation: "Monash Time Series Forecasting Archive (Godahewa et al., NeurIPS 2021 Datasets and Benchmarks)"
citation_url: "https://arxiv.org/abs/2105.06643"
license: "CC-BY-4.0 (Monash/Zenodo copy; original NN5 competition terms unknown)"
redistribution: "conditional"
frequency: "daily (1d)"
time_span: "791 consecutive days (TFB file dates 1996-03-18 to 1998-05-17 are nominal)"
length: 791
channels: 111
channel_kind: "series"
target: "OT (last series)"
missing_values: "the TFB file is the imputed version (no NaN); raw NN5 contains missing values and zeros"
protocol: "TSFLab: chronological 7:1:2 split, scaling fitted on the training split only; lookback 36/104, horizons 24/36/48/60"
literature_protocol: "TFB multivariate: lookback 36 or 104, horizons 24/36/48/60, chronological 7:1:2; the original competition used horizon 56 days"
seq_lens: [36, 104]
pred_lens: [24, 36, 48, 60]
split: "7:1:2"
stats_basis: "source-reported"
related: ["ili", "fred_md", "gift_eval/restaurant"]
config: "configs/datasets/nn5.toml"
loader: "custom"
alias: "nn5"
task_modes: ["time_series"]
---

# nn5

## Overview

NN5 is the daily cash-withdrawal history of 111 ATMs from the NN5 forecasting competition. The Monash archive republished it and TFB (Qiu et al., 2024) packaged it as a 111-channel multivariate CSV. The series have a strong day-of-week pattern, occasional holidays and outages, and a short length of 791 days, so it tests small-data seasonal forecasting.

## Provenance and license

- Competition data by Crone et al. (NN5 competition); republished by the Monash Time Series Forecasting Archive (https://zenodo.org/records/3889740, doi 10.5281/zenodo.3889740) under CC BY 4.0.
- TFB packaging: TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods (Qiu et al., PVLDB 2024), https://arxiv.org/abs/2403.20150; the TFB repository is MIT-licensed code with no separate data license.
- The original competition owner's terms were not verified, so treat redistribution as conditional on the Monash CC BY 4.0 copy.
- This preset points to `dataset/NN5/NN5.csv`, which is not shipped.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 791 daily steps | source-reported (TFB data file) |
| Channels | 111 ATM series | source-reported |
| Dates | file dates 1996-03-18 to 1998-05-17 are nominal; the competition covers two years of daily data | source-reported |
| Missing values | none in the TFB file (imputed 'without missing values' version) | source-reported |

The repository neither ships nor pins this file (`dataset/` is local and the Hub has no published copy), so the numbers above are those of the standard public distribution and a different copy may differ. Run `tsf data inspect --config configs/datasets/nn5.toml` on your copy before relying on them.

## Standard protocol and known pitfalls

- **Protocol.** TFB multivariate: horizons 24/36/48/60, lookback 36 or 104, split 7:1:2. The competition's own task was a 56-day horizon; results are not comparable.
- **Imputation.** The file hides missing values by imputation, so models are scored on filled values at some timestamps.
- **Length.** 791 steps with a 7:1:2 split leaves about 160 test rows; with lookback 104 and horizon 60 only a handful of test windows exist.
- **`drop_last`.** Loaders keep the last partial batch for every split.
- **Target column.** `features = "S"` forecasts `OT`; `"MS"` forecasts the last column.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `custom`
- Config: [`configs/datasets/nn5.toml`](../../../configs/datasets/nn5.toml)
- Local path: `./dataset/NN5/NN5.csv`
- Dataset id: `(not applicable)`
- Track: `standard`

## Input and output contract

Each item provides history/target windows and timestamp marks `(year, month, day, weekday, hour, minute)` parsed from the file's date column; after batching, values use `[batch, time, channels]`.

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
  "target": "OT"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf data inspect --config configs/datasets/nn5.toml`. Not published and not downloadable with TSFLab: place the data file at the local path above (`./dataset/NN5/NN5.csv`) yourself; `tsf data prepare` has no converter for it. Reference this preset from an experiment configuration rather than duplicating its loader parameters.

## Composition constraints

Task modes: `time_series`. Match one of them to the model's declared task mode; the
config loader rejects other combinations. Change feature or scaling parameters
only after reading the loader. Paths are repository defaults and may need local
overrides; the card does not imply that the data is bundled.
<!-- dataset-card:canonical:end -->

## Related datasets

- [`ili`](../ili/README.md): other short low-frequency set
- [`fred_md`](../fred_md/README.md): other short low-frequency set
- [`gift_eval/restaurant`](../gift_eval/restaurant/README.md): daily visitor counts at restaurants (GIFT-Eval)

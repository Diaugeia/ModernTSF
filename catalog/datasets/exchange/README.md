---
name: "exchange"
kind: "dataset"
summary: "Daily exchange rates of eight currencies (Australia, UK, Canada, Switzerland, China, Japan, New Zealand, Singapore) from the LSTNet collection, a small non-seasonal LTSF benchmark."
domain: "Finance / exchange rates"
tags: ["exchange-rate", "currency", "finance", "forex", "daily", "ltsf", "benchmark", "multivariate", "lstnet", "non-stationary", "random-walk"]
source: "Lai et al. (LSTNet) collection of daily exchange rates"
source_url: "https://github.com/laiguokun/multivariate-time-series-data"
citation: "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks (Lai et al., SIGIR 2018)"
citation_url: "https://arxiv.org/abs/1703.07015"
license: "unknown"
redistribution: "unknown"
frequency: "daily (1d)"
time_span: "1990 to 2016 per LSTNet; the standard file's date column ends 2010-09-09, so its dates are nominal"
length: 7588
channels: 8
channel_kind: "channels"
target: "OT (last column)"
missing_values: "not stated by the source; none in the standard file"
protocol: "TSFLab: chronological 7:1:2 split, scaling fitted on the training split only; lookback 96, horizons 96/192/336/720"
literature_protocol: "LTSF: lookback 96, horizons 96/192/336/720, chronological 7:1:2, MSE/MAE on z-scored data"
seq_lens: [96]
pred_lens: [96, 192, 336, 720]
split: "7:1:2"
stats_basis: "source-reported"
related: ["ili", "etth1"]
config: "configs/datasets/exchange.toml"
loader: "custom"
alias: "exchange"
task_modes: ["time_series"]
---

# exchange

## Overview

Exchange-rate holds the daily exchange rates of eight countries' currencies, collected by Lai et al. for LSTNet and reused by Autoformer and Time-Series-Library. It is tiny (8 channels, about 7.6 thousand days), has no seasonality, and behaves like correlated random walks, so it is the standard check of whether a model beats a naive last-value forecast.

## Provenance and license

- Packaging: Lai et al. 2018 (LSTNet), https://github.com/laiguokun/multivariate-time-series-data. Checked: the GitHub API reports no license, and the README only says "the collection of the daily exchange rates of eight foreign countries ... ranging from 1990 to 2016" without naming the rate provider or terms. With no original provider to read terms from, `license` and `redistribution` stay `unknown`.
- Cite: Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks (Lai et al., SIGIR 2018), https://arxiv.org/abs/1703.07015.
- This preset points to `dataset/exchange_rate/exchange_rate.csv`, which is not shipped.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows (standard file) | 7,588 daily steps | source-reported (TFB data file) |
| Channels | 8 currencies; last column `OT` is the target convention | source-reported |
| Span | LSTNet: 1990 to 2016; file dates 1990-01-01 to 2010-09-09 (7,588 consecutive days) | source-reported (conflicting; dates nominal) |
| Missing values | none in the standard file | source-reported |

The repository neither ships nor pins this file (`dataset/` is local and the Hub has no published copy), so the numbers above are those of the standard public distribution and a different copy may differ. Run `tsf data inspect --config configs/datasets/exchange.toml` on your copy before relying on them.

## Standard protocol and known pitfalls

- **Protocol.** Autoformer/TFB use 7:1:2, lookback 96, horizons 96/192/336/720 (LSTNet used 6:2:2 with horizons 3/6/12/24 days).
- **No seasonality.** Series are close to random walks, so a repeat-last-value forecast is hard to beat at long horizons; compare against it before claiming progress, and expect model rankings to differ from seasonal sets.
- **Distribution shift.** Levels drift across train and test, which z-scoring on the training split does not remove; instance normalization helps.
- **`drop_last`.** Loaders keep the last partial batch for every split.
- **Target column.** `features = "S"` forecasts `OT`; `"MS"` forecasts the last column.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `custom`
- Config: [`configs/datasets/exchange.toml`](../../../configs/datasets/exchange.toml)
- Local path: `./dataset/exchange_rate/exchange_rate.csv`
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

Inspect availability with `tsf data inspect --config configs/datasets/exchange.toml`. Not published and not downloadable with TSFLab: place the data file at the local path above (`./dataset/exchange_rate/exchange_rate.csv`) yourself; `tsf data prepare` has no converter for it. Reference this preset from an experiment configuration rather than duplicating its loader parameters.

## Composition constraints

Task modes: `time_series`. Match one of them to the model's declared task mode; the
config loader rejects other combinations. Change feature or scaling parameters
only after reading the loader. Paths are repository defaults and may need local
overrides; the card does not imply that the data is bundled.
<!-- dataset-card:canonical:end -->

## Related datasets

- [`ili`](../ili/README.md): other small non-stationary LTSF set
- [`etth1`](../etth1/README.md): standard seasonal LTSF set

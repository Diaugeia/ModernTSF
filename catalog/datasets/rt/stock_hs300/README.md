---
name: "rt/stock_hs300"
kind: "dataset"
summary: "Daily log returns of the CSI-300 constituents on business days from 2019-01-02, as a frozen release of the real-time track stock_hs300; seq_len 20 trading days, horizon 5."
domain: "Finance / equities"
tags: ["stock", "csi-300", "hs300", "china", "equities", "log-return", "daily", "real-time", "multivariate", "business-days"]
source: "Daily forward-adjusted closes fetched with AKShare (third-party Chinese market-data sites), converted to log returns"
source_url: "https://github.com/akfamily/akshare"
citation: "n/a (no dataset paper; TSFLab real-time track)"
citation_url: "n/a"
license: "unknown"
redistribution: "unknown"
frequency: "business-daily (1b)"
time_span: "2019-01-02 onward (grows with each weekly release)"
length: "grows weekly; bootstrap history starts 2019-01-02"
channels: "about 300 (CSI-300 constituents at bootstrap; the list is in the store manifest)"
channel_kind: "series"
target: "all constituents' log returns"
missing_values: "exchange holidays and suspended stocks are unobserved cells; the loader forward-fills them"
protocol: "TSFLab static protocol on a frozen release: chronological 7:1:2 split, one z-score from the training rows, seq_len 20 and horizon 5 as in the real-time track; the live rolling rounds are the separate real-time evaluation"
seq_lens: [20]
pred_lens: [5]
split: "7:1:2"
stats_basis: "source-reported"
related: ["exchange", "rt/stock_nasdaq100"]
realtime_track: "stock_hs300"
config: "configs/datasets/rt/stock_hs300.toml"
loader: "realtime_panel_ts"
alias: "rt_stock_hs300"
task_modes: ["time_series"]
---

# rt/stock_hs300

## Overview

`rt_stock_hs300` freezes the CSI-300 real-time track as a static dataset. Each channel is one constituent's daily log return; the panel store is read and split chronologically 7:1:2.

## Provenance and license

- Source: the AKShare Python library (https://github.com/akfamily/akshare, MIT-licensed code) fetching daily prices from third-party vendors (Eastmoney, Sina Finance, and others); AKShare states its data are for academic research. The per-endpoint vendor was not traced.
- Data license: not stated; MIT covers AKShare's code, not the market data, so `license` and `redistribution` are `unknown`.
- Releases are mirrored to the Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`stock_hs300/`); check vendor terms before publishing one.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Transform | log(close_t / close_{t-1}) of forward-adjusted closes | source-reported (config) |
| Bootstrap | full history from 2019-01-02 for the current constituents | source-reported (config) |
| Track config | `configs/realtime/stock_hs300.toml`: freq B, seq_len 20, horizon 5, min_coverage 0.8, timezone Asia/Shanghai | source-reported (config) |

No local copy of `dataset/realtime/stock_hs300` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

## Standard protocol and known pitfalls

- **Survivorship bias.** The panel is built from the constituents at bootstrap, so the history contains only stocks that were in the index then; static results look better than live ones.
- **Returns, not prices.** Values are daily log returns with a near-zero mean and heavy tails; a zero forecast is a strong baseline and MSE is dominated by volatile days.
- **Holidays and suspensions.** Exchange holidays and halts are unobserved cells on a business-day grid; the loader forward-fills them, which turns a missing return into a repeated one rather than a zero.
- **Frozen snapshot, not the live track.** No `version` is pinned, so the preset reads whatever the local panel store holds (pin a release, or set `revision`, for a reproducible study); the live track keeps growing and its rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation. A static split of the snapshot says nothing about data that arrive after it.
- **Gap filling.** Unobserved cells are forward-filled and only a series' leading gap is back-filled from its first reading; the loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training. Fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `realtime_panel_ts`
- Config: [`configs/datasets/rt/stock_hs300.toml`](../../../../configs/datasets/rt/stock_hs300.toml)
- Local path: `./dataset/realtime/stock_hs300`
- Dataset id: `(not applicable)`
- Track: `realtime`

## Input and output contract

Each item contains history/future values shaped `[time, nodes]` (nodes become channels) plus timestamp marks before batching; covariates are dropped.

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
  "track": "stock_hs300"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf data inspect --config configs/datasets/rt/stock_hs300.toml`; fetch
published files with `tsf data download rt/stock_hs300` when the preset is
listed by `tsf data download --list`, otherwise place the data at the local
path above (see `tsf data prepare --help`). Reference this preset from an
experiment configuration rather than duplicating its loader parameters.

## Composition constraints

Task modes: `time_series`. Match one of them to the model's declared task mode; the
config loader rejects other combinations. Change feature or scaling parameters
only after reading the loader. Paths are repository defaults and may need local
overrides; the card does not imply that the data is bundled.
<!-- dataset-card:canonical:end -->

## Related datasets

- [`exchange`](../../exchange/README.md): other finance set, static
- [`rt/stock_nasdaq100`](../stock_nasdaq100/README.md): US large-cap equivalent

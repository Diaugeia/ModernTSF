---
name: "stock_hs300"
kind: "dataset"
summary: "Daily log returns of the CSI-300 constituents on business days from 2019-01-02, the first TSFLab real-time benchmark: rounds open weekly and forecast five trading days from a 20-day history."
domain: "Finance / equities"
tags: ["stock", "csi-300", "hs300", "china", "equities", "log-return", "daily", "real-time", "leaderboard", "akshare", "multivariate", "business-days"]
source: "Daily forward-adjusted closes fetched with AKShare (third-party Chinese market-data sites)"
source_url: "https://github.com/akfamily/akshare"
citation: "n/a (no dataset paper; TSFLab real-time track stock_hs300)"
citation_url: "n/a"
license: "unknown"
redistribution: "unknown"
frequency: "business-daily (1b)"
time_span: "2019-01-02 onward (grows with each weekly round)"
length: "grows weekly; bootstrap history starts 2019-01-02"
channels: "about 300 (CSI-300 constituents at bootstrap; the exact list is in the real-time store manifest)"
channel_kind: "series"
target: "all constituents' log returns"
missing_values: "exchange holidays and suspended stocks are unobserved cells; a round is scored once 80% of target cells are observed"
protocol: "Real-time: seq_len 20 trading days, horizon 5, rolling weekly rounds, seasonal period 5, no fixed train/test split ratio"
seq_lens: [20]
pred_lens: [5]
split: "rolling weekly rounds (no split ratio)"
stats_basis: "source-reported"
related: ["exchange", "ultratraffic_ba_ts"]
realtime_track: "stock_hs300"
config: "configs/datasets/stock_hs300.toml"
loader: "cauair_ts"
alias: "stock_hs300"
task_modes: ["time_series"]
---

# stock_hs300

## Overview

`stock_hs300` is the CSI-300 real-time track as a static preset. Each channel is one index constituent's daily log return, `log(close_t / close_{t-1})`, from forward-adjusted closes. Rounds open weekly and the target is the next five trading days, which do not exist when forecasts are made (see `docs/en/realtime.md`). The preset reuses the flat-channel `cauair_ts` loader on a prepared panel under `dataset/stock_hs300`; what makes it a distinct benchmark is the alias and the `realtime` track.

## Provenance and license

- Source: the AKShare Python library (https://github.com/akfamily/akshare, MIT-licensed code) fetching daily prices; AKShare's README acknowledges third-party vendors (Eastmoney, Sina Finance, and others) and states that its data are for academic research. The per-endpoint vendor was not traced.
- Data license: not stated. MIT covers AKShare's code, not the market data, and vendor redistribution terms are unknown, so `license` and `redistribution` are `unknown`.
- Real-time data releases are mirrored to the Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`stock_hs300/`); check the vendor terms before publishing a release.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Track config | `configs/realtime/stock_hs300.toml`: freq B, seq_len 20, horizon 5, min_coverage 0.8, timezone Asia/Shanghai | source-reported (config) |
| Bootstrap | full history from 2019-01-02 for the current constituents | source-reported (config) |
| Transform | log return of forward-adjusted close | source-reported (config) |

No local copy of `dataset/stock_hs300` exists in a development checkout, so nothing is measured; length and channel count are not recorded.

## Standard protocol and known pitfalls

- **Survivorship bias.** The bootstrap uses the current constituents, so history includes only stocks that are in the index today; backtests on the history look better than live results.
- **Returns, not prices.** Values are log returns with a near-zero mean and heavy tails; a zero forecast is a strong baseline, and MSE is dominated by volatile days.
- **Holidays and suspensions.** Exchange holidays and trading halts surface as unobserved cells, not zeros; the loader reads a prepared panel, so check how gaps were filled.
- **Real-time protocol.** Weekly rounds forecast five trading days from a 20-day history and are scored once 80% of target cells are observed; this differs from static LTSF splits.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `cauair_ts`
- Config: [`configs/datasets/stock_hs300.toml`](../../../configs/datasets/stock_hs300.toml)
- Local path: `./dataset/stock_hs300`
- Dataset id: `(not applicable)`
- Track: `realtime`

## Input and output contract

Each item contains history/future values shaped `[time, nodes]` (nodes become channels) plus timestamp marks before batching; covariates are dropped.

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

Inspect availability with `tsf dataset inspect --config configs/datasets/stock_hs300.toml`; fetch
published files with `tsf dataset download stock_hs300` when the preset is
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

- [`exchange`](../exchange/README.md): other finance set, static
- [`ultratraffic_ba_ts`](../ultratraffic_ba_ts/README.md): other real-time-backed preset (traffic)

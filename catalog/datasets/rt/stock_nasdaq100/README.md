---
name: "rt/stock_nasdaq100"
kind: "dataset"
summary: "Daily log returns of the NASDAQ-100 constituents on business days from 2019-01-02, as a frozen release of the real-time track stock_nasdaq100; seq_len 20 trading days, horizon 5."
domain: "Finance / equities"
tags: ["stock", "nasdaq-100", "us", "equities", "log-return", "daily", "real-time", "multivariate", "business-days"]
source: "Split-adjusted daily closes from the Nasdaq historical API (Yahoo chart API, then Sina, as fallbacks), converted to log returns"
source_url: "https://www.nasdaq.com/market-activity/quotes/historical"
citation: "n/a (no dataset paper; TSFLab real-time track)"
citation_url: "n/a"
license: "unknown"
redistribution: "unknown"
frequency: "business-daily (1b)"
time_span: "2019-01-02 onward (grows with each weekly release)"
length: "grows weekly; bootstrap history starts 2019-01-02"
channels: "about 100 (NASDAQ-100 constituents at bootstrap; the list is in the store manifest)"
channel_kind: "series"
target: "all constituents' log returns"
missing_values: "exchange holidays and unlisted periods are unobserved cells; the loader forward-fills them"
protocol: "TSFLab static protocol on a frozen release: chronological 7:1:2 split, one z-score from the training rows, seq_len 20 and horizon 5 as in the real-time track; the live rolling rounds are the separate real-time evaluation"
seq_lens: [20]
pred_lens: [5]
split: "7:1:2"
stats_basis: "source-reported"
related: ["rt/stock_sp500", "rt/stock_hs300"]
realtime_track: "stock_nasdaq100"
config: "configs/datasets/rt/stock_nasdaq100.toml"
loader: "realtime_panel_ts"
alias: "rt_stock_nasdaq100"
task_modes: ["time_series"]
---

# rt/stock_nasdaq100

## Overview

`rt_stock_nasdaq100` freezes the NASDAQ-100 real-time track as a static dataset: each channel is one constituent's daily log return, read from the panel store and split chronologically 7:1:2.

## Provenance and license

- Source: the Nasdaq historical price API, with the Yahoo chart API and Sina as fallbacks (`docs/en/realtime.md`). Nasdaq and Yahoo return split-adjusted, not dividend-adjusted, closes; the Sina fallback is dividend-adjusted and can differ by the dividend yield on ex-dividend days.
- Vendor terms for redistribution were not verified, so `license` and `redistribution` are `unknown`.
- Releases are mirrored to the Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`stock_nasdaq100/`).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Transform | log(close_t / close_{t-1}) of split-adjusted closes | source-reported (config) |
| Bootstrap | full history from 2019-01-02 for the current constituents | source-reported (config) |
| Track config | `configs/realtime/stock_nasdaq100.toml`: freq B, seq_len 20, horizon 5, min_coverage 0.8, timezone America/New_York | source-reported (config) |

No local copy of `dataset/realtime/stock_nasdaq100` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

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
- Config: [`configs/datasets/rt/stock_nasdaq100.toml`](../../../../configs/datasets/rt/stock_nasdaq100.toml)
- Local path: `./dataset/realtime/stock_nasdaq100`
- Dataset id: `(not applicable)`
- Track: `realtime`

## Input and output contract

Each item contains history/future values shaped `[time, channels]` plus timestamp marks `(year, month, day, weekday, hour, minute)` from the panel index (all zeros when `calendar = false`).

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
  "track": "stock_nasdaq100"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf data inspect --config configs/datasets/rt/stock_nasdaq100.toml`. Not published as files. The panel store is written by the real-time track (see `docs/en/realtime.md` and `tsf realtime --help`) under the local path above, or pulled from a Hub revision via `dataset.params.revision`. Reference this preset from an experiment configuration rather than duplicating its loader parameters.

## Composition constraints

Task modes: `time_series`. Match one of them to the model's declared task mode; the
config loader rejects other combinations. Change feature or scaling parameters
only after reading the loader. Paths are repository defaults and may need local
overrides; the card does not imply that the data is bundled.
<!-- dataset-card:canonical:end -->

## Related datasets

- [`rt/stock_sp500`](../stock_sp500/README.md): broader US universe
- [`rt/stock_hs300`](../stock_hs300/README.md): China equivalent

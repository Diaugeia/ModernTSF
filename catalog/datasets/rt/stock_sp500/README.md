---
name: "rt/stock_sp500"
kind: "dataset"
summary: "Daily log returns of the S&P 500 constituents on business days from 2019-01-02, as a frozen release of the real-time track stock_sp500; seq_len 20 trading days, horizon 5."
domain: "Finance / equities"
tags: ["stock", "s&p-500", "us", "equities", "log-return", "daily", "real-time", "multivariate", "business-days"]
source: "Constituent list from datasets/s-and-p-500-companies (Wikipedia fallback); split-adjusted daily closes from the Nasdaq API, converted to log returns"
source_url: "https://github.com/datasets/s-and-p-500-companies"
citation: "n/a (no dataset paper; TSFLab real-time track)"
citation_url: "n/a"
license: "unknown"
redistribution: "unknown"
frequency: "business-daily (1b)"
time_span: "2019-01-02 onward (grows with each weekly release)"
length: "grows weekly; bootstrap history starts 2019-01-02"
channels: "about 500 (S&P 500 constituents at bootstrap; the list is in the store manifest)"
channel_kind: "series"
target: "all constituents' log returns"
missing_values: "exchange holidays and unlisted periods are unobserved cells; the loader forward-fills them"
protocol: "TSFLab static protocol on a frozen release: chronological 7:1:2 split, one z-score from the training rows, seq_len 20 and horizon 5 as in the real-time track; the live rolling rounds are the separate real-time evaluation"
seq_lens: [20]
pred_lens: [5]
split: "7:1:2"
stats_basis: "source-reported"
related: ["rt/stock_nasdaq100", "rt/stock_hs300"]
realtime_track: "stock_sp500"
config: "configs/datasets/rt/stock_sp500.toml"
loader: "realtime_panel_ts"
alias: "rt_stock_sp500"
task_modes: ["time_series"]
---

# rt/stock_sp500

## Overview

`rt_stock_sp500` freezes the S&P 500 real-time track as a static dataset: the widest stock panel in the repository, one daily log-return channel per constituent, read from the panel store at a fixed release and split chronologically 7:1:2.

## Provenance and license

- Source: the constituent list comes from `datasets/s-and-p-500-companies` (ODC-PDDL per `docs/en/realtime.md`), prices from the Nasdaq historical API as in `rt/stock_nasdaq100` (split-adjusted, not dividend-adjusted).
- Price vendor terms were not verified, so `license` and `redistribution` are `unknown` even though the list is public-domain dedicated.
- Releases are mirrored to the Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`stock_sp500/`).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Transform | log(close_t / close_{t-1}) of split-adjusted closes | source-reported (config) |
| Bootstrap | full history from 2019-01-02 for the current constituents | source-reported (config) |
| Track config | `configs/realtime/stock_sp500.toml`: freq B, seq_len 20, horizon 5, min_coverage 0.8, timezone America/New_York | source-reported (config) |

No local copy of `dataset/realtime/stock_sp500` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

## Standard protocol and known pitfalls

- **Survivorship bias.** The panel is built from the constituents at bootstrap, so the history contains only stocks that were in the index then; static results look better than live ones.
- **Returns, not prices.** Values are daily log returns with a near-zero mean and heavy tails; a zero forecast is a strong baseline and MSE is dominated by volatile days.
- **Holidays and suspensions.** Exchange holidays and halts are unobserved cells on a business-day grid; the loader forward-fills them, which turns a missing return into a repeated one rather than a zero.
- **Frozen snapshot, not the live track.** The preset reads the local panel store at a fixed release (`version`); the live track keeps growing and its rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation. A static split of the snapshot says nothing about data that arrive after it.
- **Gap filling.** Unobserved cells are forward-filled and only a series' leading gap is back-filled from its first reading; the loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training. Fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `realtime_panel_ts`
- Config: [`configs/datasets/rt/stock_sp500.toml`](../../../../configs/datasets/rt/stock_sp500.toml)
- Local path: `./dataset/realtime/stock_sp500`
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
  "track": "stock_sp500"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf data inspect --config configs/datasets/rt/stock_sp500.toml`; fetch
published files with `tsf data download rt/stock_sp500` when the preset is
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

- [`rt/stock_nasdaq100`](../stock_nasdaq100/README.md): large-cap subset of the US universe
- [`rt/stock_hs300`](../stock_hs300/README.md): China equivalent

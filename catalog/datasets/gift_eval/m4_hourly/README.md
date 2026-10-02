---
name: "gift_eval/m4_hourly"
kind: "dataset"
summary: "GIFT-Eval M4 Hourly at hourly frequency: 414 series of Econ/Fin data, mean length 902 steps; short-term horizon 48."
domain: "Econ/Fin"
tags: ["gift-eval", "benchmark", "zero-shot", "foundation-model", "univariate", "econ-fin", "hourly", "m4", "competition", "monash"]
source: "M4 Competition (Makridakis et al., 2018), via Monash"
source_url: "https://forecastingdata.org/"
citation: "GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation (Aksu et al., 2024)"
citation_url: "https://arxiv.org/abs/2410.10393"
license: "CC-BY-4.0"
redistribution: "allowed"
frequency: "hourly (1h)"
time_span: "varies by series; not recorded in the GIFT-Eval metadata used here"
length: 902
channels: 1
channel_kind: "channels"
target: "the single value series"
missing_values: "no missing-value flag recorded by GIFT-Eval; none documented for this id"
protocol: "GIFT-Eval short term: horizon 48, last 10% of each series is test (rolling non-overlapping windows, at most 20), context length chosen by the model, zero-shot oriented"
seq_lens: []
pred_lens: [48]
split: "per series: last 10% test, one horizon validation, rest train"
stats_basis: "source-reported"
related: ["gift_eval/m4_daily", "gift_eval/m4_monthly", "gift_eval/m4_quarterly", "gift_eval/m4_weekly"]
config: "configs/datasets/gift_eval/m4_hourly.toml"
loader: "gift_eval"
alias: "gift_eval/m4_hourly"
task_modes: ["time_series"]
---

# gift_eval/m4_hourly

## Overview

M4 Hourly (m4_hourly) is the GIFT-Eval series collection built from hourly series from the M4 forecasting competition (Makridakis et al.), via the Monash repository. It is part of the Econ/Fin domain of the GIFT-Eval benchmark, a zero-shot-oriented suite for general time series forecasting models (Salesforce AI Research). This preset forecasts the short-term horizon (48 steps); GIFT-Eval defines medium and long terms for some datasets, but this repository ships the short-term preset only.

## Provenance and license

- Original source: M4 Competition (Makridakis et al., 2018), via Monash; https://forecastingdata.org/.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- License of the underlying data: `CC-BY-4.0` (explicit license found for the source).
- Bytes are not bundled; download them with `tsf dataset gift-download`.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 414 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 1 | source-reported |
| Mean length per series | 902 | source-reported |
| Total observations | 373,372 | source-reported |
| Frequency | hourly (1h) | source-reported |
| Short-term test windows | 2 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Standard protocol and known pitfalls

- Split (loader behavior): per series the last `pred_len * windows` steps are test, the preceding `pred_len` steps validation, the rest train. `windows` is `min(max(1, ceil(0.1 * shortest_series_length / pred_len)), 20)` computed from the shortest series (M4 ids use 1), so it can differ from the per-dataset window counts in the benchmark paper.
- Scaling: one StandardScaler per channel is fitted on the concatenated training regions of all series, not per series. Series shorter than `seq_len + pred_len` are skipped in train/val; test windows with too little context are dropped.
- Context length is not fixed by GIFT-Eval; pick `seq_len` per model and report it. Leaderboard metrics (MASE, CRPS) are scale-free and probabilistic; scores from this preset are comparable only when the same metric and windows are used.
- Leakage: the pretraining corpora of several foundation models overlap GIFT-Eval test data (source-reported). Use the GIFT-Eval pretrain split or a clean corpus when claiming zero-shot results.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `gift_eval`
- Config: [`configs/datasets/gift_eval/m4_hourly.toml`](../../../../configs/datasets/gift_eval/m4_hourly.toml)
- Local path: `./dataset/gift_eval`
- Dataset id: `m4_hourly`
- Track: `standard`

## Input and output contract

Windowed history/target values and timestamp marks; after batching, values use `[batch, time, channels]`.

Sequence length, label length, feature mode, and batch size are supplied by the
experiment task unless explicitly overridden below.

## Dataset parameters

```json
{
  "scale": true
}
```

## Task overrides

```json
{
  "features": "M",
  "pred_len": 48
}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/gift_eval/m4_hourly.toml`; fetch
published files with `tsf dataset download gift_eval/m4_hourly` when the preset is
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

- [`gift_eval/m4_daily`](../m4_daily/README.md)
- [`gift_eval/m4_monthly`](../m4_monthly/README.md)
- [`gift_eval/m4_quarterly`](../m4_quarterly/README.md)
- [`gift_eval/m4_weekly`](../m4_weekly/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.

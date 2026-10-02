---
name: "gift_eval/car_parts_with_missing"
kind: "dataset"
summary: "GIFT-Eval Car Parts at monthly frequency: 2,674 series of Sales data, mean length 51 steps; short-term horizon 12."
domain: "Sales"
tags: ["gift-eval", "benchmark", "zero-shot", "foundation-model", "univariate", "sales", "monthly", "sales", "intermittent", "monash"]
source: "Car Parts (Hyndman et al.), via the Monash Time Series Forecasting Repository"
source_url: "https://zenodo.org/records/4656022"
citation: "GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation (Aksu et al., 2024)"
citation_url: "https://arxiv.org/abs/2410.10393"
license: "CC-BY-4.0"
redistribution: "allowed"
frequency: "monthly (1mo)"
time_span: "varies by series; not recorded in the GIFT-Eval metadata used here"
length: 51
channels: 1
channel_kind: "channels"
target: "the single value series"
missing_values: "contains missing values (NaN); the loader passes them through"
protocol: "GIFT-Eval short term: horizon 12, last 10% of each series is test (rolling non-overlapping windows, at most 20), context length chosen by the model, zero-shot oriented"
seq_lens: []
pred_lens: [12]
split: "per series: last 10% test, one horizon validation, rest train"
stats_basis: "source-reported"
related: []
config: "configs/datasets/gift_eval/car_parts_with_missing.toml"
loader: "gift_eval"
alias: "gift_eval/car_parts_with_missing"
task_modes: ["time_series"]
---

# gift_eval/car_parts_with_missing

## Overview

Car Parts (car_parts_with_missing) is the GIFT-Eval series collection built from monthly sales of car parts with many missing (intermittent) entries, from the Monash repository. It is part of the Sales domain of the GIFT-Eval benchmark, a zero-shot-oriented suite for general time series forecasting models (Salesforce AI Research). This preset forecasts the short-term horizon (12 steps); GIFT-Eval defines medium and long terms for some datasets, but this repository ships the short-term preset only.

## Provenance and license

- Original source: Car Parts (Hyndman et al.), via the Monash Time Series Forecasting Repository; https://zenodo.org/records/4656022.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- License of the underlying data: `CC-BY-4.0` (explicit license found for the source).
- Bytes are not bundled; download them with `tsf data prepare --from gift`.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 2,674 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 1 | source-reported |
| Mean length per series | 51 | source-reported |
| Total observations | 136,374 | source-reported |
| Frequency | monthly (1mo) | source-reported |
| Short-term test windows | 1 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Standard protocol and known pitfalls

- Split (loader behavior): per series the last `pred_len * windows` steps are test, the preceding `pred_len` steps validation, the rest train. `windows` is `min(max(1, ceil(0.1 * shortest_series_length / pred_len)), 20)` computed from the shortest series (M4 ids use 1), so it can differ from the per-dataset window counts in the benchmark paper.
- Scaling: one StandardScaler per channel is fitted on the concatenated training regions of all series, not per series. Series shorter than `seq_len + pred_len` are skipped in train/val; test windows with too little context are dropped.
- Context length is not fixed by GIFT-Eval; pick `seq_len` per model and report it. Leaderboard metrics (MASE, CRPS) are scale-free and probabilistic; scores from this preset are comparable only when the same metric and windows are used.
- Leakage: the pretraining corpora of several foundation models overlap GIFT-Eval test data (source-reported). Use the GIFT-Eval pretrain split or a clean corpus when claiming zero-shot results.
- Missing values: NaNs pass through the loader (scaler statistics ignore them), so models receive NaN inputs and targets unless you impute or mask them.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `gift_eval`
- Config: [`configs/datasets/gift_eval/car_parts_with_missing.toml`](../../../../configs/datasets/gift_eval/car_parts_with_missing.toml)
- Local path: `./dataset/gift_eval`
- Dataset id: `car_parts_with_missing`
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
  "pred_len": 12
}
```

## Preparation and use

Inspect availability with `tsf data inspect --config configs/datasets/gift_eval/car_parts_with_missing.toml`; fetch
published files with `tsf data download gift_eval/car_parts_with_missing` when the preset is
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

- [`gift_eval`](../README.md): the GIFT-Eval family card.

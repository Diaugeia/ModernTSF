---
name: "gift_eval/covid_deaths"
kind: "dataset"
summary: "GIFT-Eval COVID Deaths at daily frequency: 266 series of Healthcare data, mean length 212 steps; short-term horizon 30."
domain: "Healthcare"
tags: ["gift-eval", "benchmark", "zero-shot", "foundation-model", "univariate", "healthcare", "daily", "covid-19", "epidemiology", "monash"]
source: "Monash Time Series Forecasting Repository (Godahewa et al., 2021)"
source_url: "https://zenodo.org/records/4656009"
citation: "GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation (Aksu et al., 2024)"
citation_url: "https://arxiv.org/abs/2410.10393"
license: "CC-BY-4.0"
redistribution: "allowed"
frequency: "daily (1d)"
time_span: "varies by series; not recorded in the GIFT-Eval metadata used here"
length: 212
channels: 1
channel_kind: "channels"
target: "the single value series"
missing_values: "no missing-value flag recorded by GIFT-Eval; none documented for this id"
protocol: "GIFT-Eval short term: horizon 30, last 10% of each series is test (rolling non-overlapping windows, at most 20), context length chosen by the model, zero-shot oriented"
seq_lens: []
pred_lens: [30]
split: "per series: last 10% test, one horizon validation, rest train"
stats_basis: "source-reported"
related: []
config: "configs/datasets/gift_eval/covid_deaths.toml"
loader: "gift_eval"
alias: "gift_eval/covid_deaths"
task_modes: ["time_series"]
---

# gift_eval/covid_deaths

## Overview

COVID Deaths (covid_deaths) is the GIFT-Eval series collection built from daily cumulative COVID-19 deaths per country or region, from the Monash repository. It is part of the Healthcare domain of the GIFT-Eval benchmark, a zero-shot-oriented suite for general time series forecasting models (Salesforce AI Research). This preset forecasts the short-term horizon (30 steps); GIFT-Eval defines medium and long terms for some datasets, but this repository ships the short-term preset only.

## Provenance and license

- Original source: Monash Time Series Forecasting Repository (Godahewa et al., 2021); https://zenodo.org/records/4656009 (index: https://forecastingdata.org/).
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- License of the underlying data: `CC-BY-4.0` (explicit license found for the source).
- Bytes are not bundled; download them with `tsf data prepare --from gift`.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 266 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 1 | source-reported |
| Mean length per series | 212 | source-reported |
| Total observations | 56,392 | source-reported |
| Frequency | daily (1d) | source-reported |
| Short-term test windows | 1 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Standard protocol and known pitfalls

- Split (loader behavior): per series the last `pred_len * windows` steps are test, the preceding `pred_len` steps validation, the rest train. `windows` is `min(max(1, ceil(0.1 * shortest_series_length / pred_len)), 20)` computed from the shortest series (M4 ids use 1), so it can differ from the per-dataset window counts in the benchmark paper.
- Scaling: one StandardScaler per channel is fitted on the concatenated training regions of all series, not per series. Series shorter than `seq_len + pred_len` are skipped in train/val; test windows with too little context are dropped.
- Context length is not fixed by GIFT-Eval; pick `seq_len` per model and report it. Leaderboard metrics (MASE, CRPS) are scale-free and probabilistic; scores from this preset are comparable only when the same metric and windows are used.
- Leakage: the pretraining corpora of several foundation models overlap GIFT-Eval test data (source-reported). Use the GIFT-Eval pretrain split or a clean corpus when claiming zero-shot results.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `gift_eval`
- Config: [`configs/datasets/gift_eval/covid_deaths.toml`](../../../../configs/datasets/gift_eval/covid_deaths.toml)
- Local path: `./dataset/gift_eval`
- Dataset id: `covid_deaths`
- Track: `standard`

## Input and output contract

Windowed history/target values and timestamp marks `(year, month, day, weekday, hour, minute)` synthesised from each series' start date and frequency; after batching, values use `[batch, time, channels]`.

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
  "pred_len": 30
}
```

## Preparation and use

Inspect availability with `tsf data inspect --config configs/datasets/gift_eval/covid_deaths.toml`. Not published through the TSFLab manifest. Fetch the Hugging Face `Salesforce/GiftEval` data with `tsf data prepare --from gift --datasets covid_deaths` (or `--link-only` for an existing download); it links `./dataset/gift_eval`. Reference this preset from an experiment configuration rather than duplicating its loader parameters.

## Composition constraints

Task modes: `time_series`. Match one of them to the model's declared task mode; the
config loader rejects other combinations. Change feature or scaling parameters
only after reading the loader. Paths are repository defaults and may need local
overrides; the card does not imply that the data is bundled.
<!-- dataset-card:canonical:end -->

## Related datasets

- [`gift_eval`](../README.md): the GIFT-Eval family card.

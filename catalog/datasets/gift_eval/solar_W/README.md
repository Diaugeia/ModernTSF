---
name: "gift_eval/solar_W"
kind: "dataset"
summary: "GIFT-Eval Solar at weekly frequency: 137 series of Energy data, mean length 52 steps; short-term horizon 8."
domain: "Energy"
tags: ["gift-eval", "benchmark", "zero-shot", "foundation-model", "univariate", "energy", "weekly", "solar", "photovoltaic", "nrel"]
source: "NREL Solar Power Data for Integrated Variable Generation via LSTNet (Lai et al.)"
source_url: "https://github.com/laiguokun/multivariate-time-series-data"
citation: "GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation (Aksu et al., 2024)"
citation_url: "https://arxiv.org/abs/2410.10393"
license: "unknown"
redistribution: "unknown"
frequency: "weekly (1w)"
time_span: "2006-01-01 to 2006-12-31"
length: 52
channels: 1
channel_kind: "channels"
target: "the single value series"
missing_values: "no missing-value flag recorded by GIFT-Eval; none documented for this id"
protocol: "GIFT-Eval short term: horizon 8, last 10% of each series is test (rolling non-overlapping windows, at most 20), context length chosen by the model, zero-shot oriented"
seq_lens: []
pred_lens: [8]
split: "per series: last 10% test, one horizon validation, rest train"
stats_basis: "source-reported"
related: ["solar", "gift_eval/solar_10T", "gift_eval/solar_D", "gift_eval/solar_H"]
config: "configs/datasets/gift_eval/solar_W.toml"
loader: "gift_eval"
alias: "gift_eval/solar_W"
task_modes: ["time_series"]
---

# gift_eval/solar_W

## Overview

Solar (solar/W) is the GIFT-Eval series collection built from solar power production records of 137 photovoltaic plants in Alabama for 2006 (NREL data), as prepared for LSTNet. It is part of the Energy domain of the GIFT-Eval benchmark, a zero-shot-oriented suite for general time series forecasting models (Salesforce AI Research). This preset forecasts the short-term horizon (8 steps); GIFT-Eval defines medium and long terms for some datasets, but this repository ships the short-term preset only.

## Provenance and license

- Original source: NREL Solar Power Data for Integrated Variable Generation via LSTNet (Lai et al.); https://github.com/laiguokun/multivariate-time-series-data.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- GIFT-Eval wrapper: the Hugging Face card for `Salesforce/GiftEval` (https://huggingface.co/datasets/Salesforce/GiftEval) declares `license: apache-2.0` for the whole benchmark and lists no per-subset licenses; its Ethical Considerations say the release is "for research purposes only in support of an academic paper". The Apache-2.0 tag does not relicense the underlying data, so the terms below come from each original source.
- Underlying data terms: not stated. The NREL/NLR page (https://www.nlr.gov/grid/solar-power-data) has no license or terms, the NLR disclaimer (https://www.nlr.gov/disclaimer.html) grants none, and the LSTNet repository (https://github.com/laiguokun/multivariate-time-series-data) has no license file; `license` and `redistribution` stay `unknown`.
- Bytes are not bundled; download them with `tsf data prepare --from gift`.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 137 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 1 | source-reported |
| Mean length per series | 52 | source-reported |
| Total observations | 7,124 | source-reported |
| Frequency | weekly (1w) | source-reported |
| Short-term test windows | 1 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Standard protocol and known pitfalls

- Split (loader behavior): per series the last `pred_len * windows` steps are test, the preceding `pred_len` steps validation, the rest train. `windows` is `min(max(1, ceil(0.1 * shortest_series_length / pred_len)), 20)` computed from the shortest series (M4 ids use 1), so it can differ from the per-dataset window counts in the benchmark paper.
- Scaling: one StandardScaler per channel is fitted on the concatenated training regions of all series, not per series. Series shorter than `seq_len + pred_len` are skipped in train/val; test windows with too little context are dropped.
- Context length is not fixed by GIFT-Eval; pick `seq_len` per model and report it. Leaderboard metrics (MASE, CRPS) are scale-free and probabilistic; scores from this preset are comparable only when the same metric and windows are used.
- Leakage: the pretraining corpora of several foundation models overlap GIFT-Eval test data (source-reported). Use the GIFT-Eval pretrain split or a clean corpus when claiming zero-shot results.
- Solar output is zero at night (55% of values are zero in the local LTSF `solar` file), so relative-error metrics such as MAPE are undefined there.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `gift_eval`
- Config: [`configs/datasets/gift_eval/solar_W.toml`](../../../../configs/datasets/gift_eval/solar_W.toml)
- Local path: `./dataset/gift_eval`
- Dataset id: `solar/W`
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
  "pred_len": 8
}
```

## Preparation and use

Inspect availability with `tsf data inspect --config configs/datasets/gift_eval/solar_W.toml`. Not published through the TSFLab manifest. Fetch the Hugging Face `Salesforce/GiftEval` data with `tsf data prepare --from gift --datasets solar/W` (or `--link-only` for an existing download); it links `./dataset/gift_eval`. Reference this preset from an experiment configuration rather than duplicating its loader parameters.

## Composition constraints

Task modes: `time_series`. Match one of them to the model's declared task mode; the
config loader rejects other combinations. Change feature or scaling parameters
only after reading the loader. Paths are repository defaults and may need local
overrides; the card does not imply that the data is bundled.
<!-- dataset-card:canonical:end -->

## Related datasets

- [`solar`](../../solar/README.md)
- [`gift_eval/solar_10T`](../solar_10T/README.md)
- [`gift_eval/solar_D`](../solar_D/README.md)
- [`gift_eval/solar_H`](../solar_H/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.

---
name: "gift_eval/electricity_W"
kind: "dataset"
summary: "GIFT-Eval Electricity at weekly frequency: 370 series of Energy data, mean length 208 steps; short-term horizon 8."
domain: "Energy"
tags: ["gift-eval", "benchmark", "zero-shot", "foundation-model", "univariate", "energy", "weekly", "electricity", "load", "uci"]
source: "UCI ElectricityLoadDiagrams20112014 (Trindade, 2015)"
source_url: "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014"
citation: "GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation (Aksu et al., 2024)"
citation_url: "https://arxiv.org/abs/2410.10393"
license: "CC-BY-4.0"
redistribution: "allowed"
frequency: "weekly (1w)"
time_span: "2011-01-01 to 2014-12-31"
length: 208
channels: 1
channel_kind: "channels"
target: "the single value series"
missing_values: "no missing-value flag recorded by GIFT-Eval; none documented for this id"
protocol: "GIFT-Eval short term: horizon 8, last 10% of each series is test (rolling non-overlapping windows, at most 20), context length chosen by the model, zero-shot oriented"
seq_lens: []
pred_lens: [8]
split: "per series: last 10% test, one horizon validation, rest train"
stats_basis: "source-reported"
related: ["electricity", "gift_eval/electricity_15T", "gift_eval/electricity_D", "gift_eval/electricity_H"]
config: "configs/datasets/gift_eval/electricity_W.toml"
loader: "gift_eval"
alias: "gift_eval/electricity_W"
task_modes: ["time_series"]
---

# gift_eval/electricity_W

## Overview

Electricity (electricity/W) is the GIFT-Eval series collection built from electricity consumption of 370 Portuguese clients, 2011-2014 (UCI ElectricityLoadDiagrams20112014). It is part of the Energy domain of the GIFT-Eval benchmark, a zero-shot-oriented suite for general time series forecasting models (Salesforce AI Research). This preset forecasts the short-term horizon (8 steps); GIFT-Eval defines medium and long terms for some datasets, but this repository ships the short-term preset only.

## Provenance and license

- Original source: UCI ElectricityLoadDiagrams20112014 (Trindade, 2015); https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- License of the underlying data: `CC-BY-4.0` (explicit license found for the source).
- Bytes are not bundled; download them with `tsf data prepare --from gift`.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 370 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 1 | source-reported |
| Mean length per series | 208 | source-reported |
| Total observations | 76,960 | source-reported |
| Frequency | weekly (1w) | source-reported |
| Short-term test windows | 3 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Standard protocol and known pitfalls

- Split (loader behavior): per series the last `pred_len * windows` steps are test, the preceding `pred_len` steps validation, the rest train. `windows` is `min(max(1, ceil(0.1 * shortest_series_length / pred_len)), 20)` computed from the shortest series (M4 ids use 1), so it can differ from the per-dataset window counts in the benchmark paper.
- Scaling: one StandardScaler per channel is fitted on the concatenated training regions of all series, not per series. Series shorter than `seq_len + pred_len` are skipped in train/val; test windows with too little context are dropped.
- Context length is not fixed by GIFT-Eval; pick `seq_len` per model and report it. Leaderboard metrics (MASE, CRPS) are scale-free and probabilistic; scores from this preset are comparable only when the same metric and windows are used.
- Leakage: the pretraining corpora of several foundation models overlap GIFT-Eval test data (source-reported). Use the GIFT-Eval pretrain split or a clean corpus when claiming zero-shot results.
- 370 clients here versus the 321 clients of the LTSF `electricity` preset.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `gift_eval`
- Config: [`configs/datasets/gift_eval/electricity_W.toml`](../../../../configs/datasets/gift_eval/electricity_W.toml)
- Local path: `./dataset/gift_eval`
- Dataset id: `electricity/W`
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
  "pred_len": 8
}
```

## Preparation and use

Inspect availability with `tsf data inspect --config configs/datasets/gift_eval/electricity_W.toml`; fetch
published files with `tsf data download gift_eval/electricity_W` when the preset is
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

- [`electricity`](../../electricity/README.md)
- [`gift_eval/electricity_15T`](../electricity_15T/README.md)
- [`gift_eval/electricity_D`](../electricity_D/README.md)
- [`gift_eval/electricity_H`](../electricity_H/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.

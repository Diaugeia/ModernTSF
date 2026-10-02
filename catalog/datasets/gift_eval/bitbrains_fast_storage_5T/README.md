---
name: "gift_eval/bitbrains_fast_storage_5T"
kind: "dataset"
summary: "GIFT-Eval Bitbrains fastStorage at 5-minute frequency: 1,250 series, 2 variates each of Web/CloudOps data, mean length 8,640 steps; short-term horizon 48."
domain: "Web/CloudOps"
tags: ["gift-eval", "benchmark", "zero-shot", "foundation-model", "multivariate", "web-cloudops", "5min", "cloud", "cpu", "virtual-machines", "bitbrains"]
source: "Bitbrains traces, Grid Workloads Archive (Shen et al., 2015)"
source_url: "http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains"
citation: "GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation (Aksu et al., 2024)"
citation_url: "https://arxiv.org/abs/2410.10393"
license: "unknown"
redistribution: "unknown"
frequency: "5-minute (5min)"
time_span: "varies by series; not recorded in the GIFT-Eval metadata used here"
length: 8640
channels: 2
channel_kind: "channels"
target: "all variates"
missing_values: "no missing-value flag recorded by GIFT-Eval; none documented for this id"
protocol: "GIFT-Eval short term: horizon 48, last 10% of each series is test (rolling non-overlapping windows, at most 20), context length chosen by the model, zero-shot oriented"
seq_lens: []
pred_lens: [48]
split: "per series: last 10% test, one horizon validation, rest train"
stats_basis: "source-reported"
related: ["gift_eval/bitbrains_fast_storage_H"]
config: "configs/datasets/gift_eval/bitbrains_fast_storage_5T.toml"
loader: "gift_eval"
alias: "gift_eval/bitbrains_fast_storage_5T"
task_modes: ["time_series"]
---

# gift_eval/bitbrains_fast_storage_5T

## Overview

Bitbrains fastStorage (bitbrains_fast_storage/5T) is the GIFT-Eval series collection built from resource-usage traces of virtual machines on the fastStorage tier of a managed-hosting provider (Bitbrains). It is part of the Web/CloudOps domain of the GIFT-Eval benchmark, a zero-shot-oriented suite for general time series forecasting models (Salesforce AI Research). This preset forecasts the short-term horizon (48 steps); GIFT-Eval defines medium and long terms for some datasets, but this repository ships the short-term preset only.

## Provenance and license

- Original source: Bitbrains traces, Grid Workloads Archive (Shen et al., 2015); http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- License of the underlying data: `unknown`; no explicit terms were found, so redistribution is not assumed.
- Bytes are not bundled; download them with `tsf dataset gift-download`.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 1,250 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 2 | source-reported |
| Mean length per series | 8,640 | source-reported |
| Total observations | 10,800,000 | source-reported |
| Frequency | 5-minute (5min) | source-reported |
| Short-term test windows | 18 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Standard protocol and known pitfalls

- Split (loader behavior): per series the last `pred_len * windows` steps are test, the preceding `pred_len` steps validation, the rest train. `windows` is `min(max(1, ceil(0.1 * shortest_series_length / pred_len)), 20)` computed from the shortest series (M4 ids use 1), so it can differ from the per-dataset window counts in the benchmark paper.
- Scaling: one StandardScaler per channel is fitted on the concatenated training regions of all series, not per series. Series shorter than `seq_len + pred_len` are skipped in train/val; test windows with too little context are dropped.
- Context length is not fixed by GIFT-Eval; pick `seq_len` per model and report it. Leaderboard metrics (MASE, CRPS) are scale-free and probabilistic; scores from this preset are comparable only when the same metric and windows are used.
- Leakage: the pretraining corpora of several foundation models overlap GIFT-Eval test data (source-reported). Use the GIFT-Eval pretrain split or a clean corpus when claiming zero-shot results.
- Multivariate: `features = "M"` forecasts all variates; `S` keeps the first variate only.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `gift_eval`
- Config: [`configs/datasets/gift_eval/bitbrains_fast_storage_5T.toml`](../../../../configs/datasets/gift_eval/bitbrains_fast_storage_5T.toml)
- Local path: `./dataset/gift_eval`
- Dataset id: `bitbrains_fast_storage/5T`
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

Inspect availability with `tsf dataset inspect --config configs/datasets/gift_eval/bitbrains_fast_storage_5T.toml`; fetch
published files with `tsf dataset download gift_eval/bitbrains_fast_storage_5T` when the preset is
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

- [`gift_eval/bitbrains_fast_storage_H`](../bitbrains_fast_storage_H/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.

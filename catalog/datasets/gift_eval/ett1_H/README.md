---
name: "gift_eval/ett1_H"
kind: "dataset"
summary: "GIFT-Eval ETT (set 1) at hourly frequency: 1 series, 7 variates each of Energy data, mean length 17,420 steps; short-term horizon 48."
domain: "Energy"
tags: ["gift-eval", "benchmark", "zero-shot", "foundation-model", "multivariate", "energy", "hourly", "ett", "transformer", "oil-temperature"]
source: "ETT dataset (Zhou et al., Informer)"
source_url: "https://github.com/zhouhaoyi/ETDataset"
citation: "GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation (Aksu et al., 2024)"
citation_url: "https://arxiv.org/abs/2410.10393"
license: "CC-BY-ND-4.0"
redistribution: "conditional"
frequency: "hourly (1h)"
time_span: "2016-07-01 to 2018-06-26 (aggregated from the ETT source)"
length: 17420
channels: 7
channel_kind: "channels"
target: "all variates"
missing_values: "no missing-value flag recorded by GIFT-Eval; none documented for this id"
protocol: "GIFT-Eval short term: horizon 48, last 10% of each series is test (rolling non-overlapping windows, at most 20), context length chosen by the model, zero-shot oriented"
seq_lens: []
pred_lens: [48]
split: "per series: last 10% test, one horizon validation, rest train"
stats_basis: "source-reported"
related: ["etth1", "ettm1", "gift_eval/ett1_15T", "gift_eval/ett1_D", "gift_eval/ett1_W"]
config: "configs/datasets/gift_eval/ett1_H.toml"
loader: "gift_eval"
alias: "gift_eval/ett1_H"
task_modes: ["time_series"]
---

# gift_eval/ett1_H

## Overview

ETT (set 1) (ett1/H) is the GIFT-Eval series collection built from oil temperature and six power-load features of the first Electricity Transformer Temperature station (ETTh1/ETTm1 lineage), resampled to this frequency. It is part of the Energy domain of the GIFT-Eval benchmark, a zero-shot-oriented suite for general time series forecasting models (Salesforce AI Research). This preset forecasts the short-term horizon (48 steps); GIFT-Eval defines medium and long terms for some datasets, but this repository ships the short-term preset only.

## Provenance and license

- Original source: ETT dataset (Zhou et al., Informer); https://github.com/zhouhaoyi/ETDataset.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- License of the underlying data: `CC-BY-ND-4.0` (explicit license found for the source).
- Bytes are not bundled; download them with `tsf dataset gift-download`.
- The no-derivatives clause matters for resampled or re-published copies; keep the original files unmodified when redistributing.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 1 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 7 | source-reported |
| Mean length per series | 17,420 | source-reported |
| Total observations | 17,420 | source-reported |
| Frequency | hourly (1h) | source-reported |
| Short-term test windows | 20 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Standard protocol and known pitfalls

- Split (loader behavior): per series the last `pred_len * windows` steps are test, the preceding `pred_len` steps validation, the rest train. `windows` is `min(max(1, ceil(0.1 * shortest_series_length / pred_len)), 20)` computed from the shortest series (M4 ids use 1), so it can differ from the per-dataset window counts in the benchmark paper.
- Scaling: one StandardScaler per channel is fitted on the concatenated training regions of all series, not per series. Series shorter than `seq_len + pred_len` are skipped in train/val; test windows with too little context are dropped.
- Context length is not fixed by GIFT-Eval; pick `seq_len` per model and report it. Leaderboard metrics (MASE, CRPS) are scale-free and probabilistic; scores from this preset are comparable only when the same metric and windows are used.
- Leakage: the pretraining corpora of several foundation models overlap GIFT-Eval test data (source-reported). Use the GIFT-Eval pretrain split or a clean corpus when claiming zero-shot results.
- Multivariate: `features = "M"` forecasts all variates; `S` keeps the first variate only.
- This is the full ETT series resampled by GIFT-Eval, not the 12/4/4-month LTSF protocol used by the `etth*`/`ettm*` presets; results are not comparable.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `gift_eval`
- Config: [`configs/datasets/gift_eval/ett1_H.toml`](../../../../configs/datasets/gift_eval/ett1_H.toml)
- Local path: `./dataset/gift_eval`
- Dataset id: `ett1/H`
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

Inspect availability with `tsf dataset inspect --config configs/datasets/gift_eval/ett1_H.toml`; fetch
published files with `tsf dataset download gift_eval/ett1_H` when the preset is
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

- [`etth1`](../../etth1/README.md)
- [`ettm1`](../../ettm1/README.md)
- [`gift_eval/ett1_15T`](../ett1_15T/README.md)
- [`gift_eval/ett1_D`](../ett1_D/README.md)
- [`gift_eval/ett1_W`](../ett1_W/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.

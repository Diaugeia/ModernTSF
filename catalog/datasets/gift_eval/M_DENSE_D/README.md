---
name: "gift_eval/M_DENSE_D"
kind: "dataset"
summary: "GIFT-Eval M-DENSE at daily frequency: 30 series of Transport data, mean length 730 steps; short-term horizon 30."
domain: "Transport"
tags: ["gift-eval", "benchmark", "zero-shot", "foundation-model", "univariate", "transport", "daily", "traffic", "madrid", "libcity"]
source: "LibCity / M-DENSE Madrid"
source_url: "https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico"
citation: "GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation (Aksu et al., 2024)"
citation_url: "https://arxiv.org/abs/2410.10393"
license: "CC-BY-4.0"
redistribution: "allowed"
frequency: "daily (1d)"
time_span: "varies by series; not recorded in the GIFT-Eval metadata used here"
length: 730
channels: 1
channel_kind: "channels"
target: "the single value series"
missing_values: "no missing-value flag recorded by GIFT-Eval; none documented for this id"
protocol: "GIFT-Eval short term: horizon 30, last 10% of each series is test (rolling non-overlapping windows, at most 20), context length chosen by the model, zero-shot oriented"
seq_lens: []
pred_lens: [30]
split: "per series: last 10% test, one horizon validation, rest train"
stats_basis: "source-reported"
related: ["gift_eval/M_DENSE_H"]
config: "configs/datasets/gift_eval/M_DENSE_D.toml"
loader: "gift_eval"
alias: "gift_eval/M_DENSE_D"
task_modes: ["time_series"]
---

# gift_eval/M_DENSE_D

## Overview

M-DENSE (M_DENSE/D) is the GIFT-Eval series collection built from urban traffic measurements from Madrid, prepared for GIFT-Eval from the LibCity collection. It is part of the Transport domain of the GIFT-Eval benchmark, a zero-shot-oriented suite for general time series forecasting models (Salesforce AI Research). This preset forecasts the short-term horizon (30 steps); GIFT-Eval defines medium and long terms for some datasets, but this repository ships the short-term preset only.

## Provenance and license

- Original source: LibCity / M-DENSE Madrid; https://github.com/LibCity/Bigscity-LibCity.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- GIFT-Eval wrapper: the Hugging Face card for `Salesforce/GiftEval` (https://huggingface.co/datasets/Salesforce/GiftEval) declares `license: apache-2.0` for the whole benchmark and lists no per-subset licenses; its Ethical Considerations say the release is "for research purposes only in support of an academic paper". The Apache-2.0 tag does not relicense the underlying data, so the terms below come from each original source.
- Underlying data terms: LibCity's M-DENSE comes from the Madrid City Council traffic-measurement history (de Medrano and Aznarte, 2020, https://arxiv.org/abs/2003.13977, which names the municipality's open data portal as the source). The portal record "Trafico. Historico de datos del trafico desde 2013" (https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico) lists the licence "Creative Commons Attribution 4.0 International (CC BY 4.0)". The link from LibCity's 30-sensor 2018-2019 extract to that portal record is by the paper's description, not a per-file statement; redistribute with attribution to Ayuntamiento de Madrid.
- Bytes are not bundled; download them with `tsf dataset gift-download`.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 30 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 1 | source-reported |
| Mean length per series | 730 | source-reported |
| Total observations | 21,900 | source-reported |
| Frequency | daily (1d) | source-reported |
| Short-term test windows | 3 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Standard protocol and known pitfalls

- Split (loader behavior): per series the last `pred_len * windows` steps are test, the preceding `pred_len` steps validation, the rest train. `windows` is `min(max(1, ceil(0.1 * shortest_series_length / pred_len)), 20)` computed from the shortest series (M4 ids use 1), so it can differ from the per-dataset window counts in the benchmark paper.
- Scaling: one StandardScaler per channel is fitted on the concatenated training regions of all series, not per series. Series shorter than `seq_len + pred_len` are skipped in train/val; test windows with too little context are dropped.
- Context length is not fixed by GIFT-Eval; pick `seq_len` per model and report it. Leaderboard metrics (MASE, CRPS) are scale-free and probabilistic; scores from this preset are comparable only when the same metric and windows are used.
- Leakage: the pretraining corpora of several foundation models overlap GIFT-Eval test data (source-reported). Use the GIFT-Eval pretrain split or a clean corpus when claiming zero-shot results.
- Series are traffic sensors without an adjacency matrix in this preset; use graph loaders only if you supply one separately.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `gift_eval`
- Config: [`configs/datasets/gift_eval/M_DENSE_D.toml`](../../../../configs/datasets/gift_eval/M_DENSE_D.toml)
- Local path: `./dataset/gift_eval`
- Dataset id: `M_DENSE/D`
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
  "pred_len": 30
}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/gift_eval/M_DENSE_D.toml`; fetch
published files with `tsf dataset download gift_eval/M_DENSE_D` when the preset is
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

- [`gift_eval/M_DENSE_H`](../M_DENSE_H/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.

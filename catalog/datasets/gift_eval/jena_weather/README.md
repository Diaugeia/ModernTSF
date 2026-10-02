---
name: "gift_eval/jena_weather"
kind: "dataset"
summary: "GIFT-Eval Jena Weather: one series of 21 meteorological variates, short-term horizon 48; frequency and length are not recorded for this id."
domain: "Nature"
tags: ["gift-eval", "benchmark", "zero-shot", "foundation-model", "multivariate", "nature", "weather", "meteorology", "jena"]
source: "MPI for Biogeochemistry, Jena weather station"
source_url: "https://www.bgc-jena.mpg.de/wetter/weather_data.html"
citation: "GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation (Aksu et al., 2024)"
citation_url: "https://arxiv.org/abs/2410.10393"
license: "CC-BY-4.0"
redistribution: "allowed"
frequency: "not recorded (the id has no frequency suffix)"
time_span: "varies by series; not recorded in the GIFT-Eval metadata used here"
length: "not reported"
channels: 21
channel_kind: "channels"
target: "all variates"
missing_values: "no missing-value flag recorded by GIFT-Eval; none documented for this id"
protocol: "GIFT-Eval short term: horizon 48, last 10% of each series is test (rolling non-overlapping windows, at most 20), context length chosen by the model, zero-shot oriented"
seq_lens: []
pred_lens: [48]
split: "per series: last 10% test, one horizon validation, rest train"
stats_basis: "source-reported"
related: ["weather"]
config: "configs/datasets/gift_eval/jena_weather.toml"
loader: "gift_eval"
alias: "gift_eval/jena_weather"
task_modes: ["time_series"]
---

# gift_eval/jena_weather

## Overview

Jena Weather (jena_weather) is the GIFT-Eval series collection built from 21 meteorological indicators from the Max Planck Institute for Biogeochemistry weather station in Jena, as prepared for Autoformer. It is part of the Nature domain of the GIFT-Eval benchmark, a zero-shot-oriented suite for general time series forecasting models (Salesforce AI Research). This preset forecasts the short-term horizon (48 steps); GIFT-Eval defines medium and long terms for some datasets, but this repository ships the short-term preset only.

## Provenance and license

- Original source: MPI for Biogeochemistry, Jena weather station; https://www.bgc-jena.mpg.de/wetter/.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- GIFT-Eval wrapper: the Hugging Face card for `Salesforce/GiftEval` (https://huggingface.co/datasets/Salesforce/GiftEval) declares `license: apache-2.0` for the whole benchmark and lists no per-subset licenses; its Ethical Considerations say the release is "for research purposes only in support of an academic paper". The Apache-2.0 tag does not relicense the underlying data, so the terms below come from each original source.
- Underlying data terms: the MPI-BGC Data Download page (https://www.bgc-jena.mpg.de/wetter/weather_data.html) states "Terms of Use (as per Creative Commons CC-BY-4.0)"; the landing page itself states none. Redistribution is allowed with attribution to the Max Planck Institute for Biogeochemistry.
- Bytes are not bundled; download them with `tsf data prepare --from gift`.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 1 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 21 | source-reported |
| Mean length per series | not reported | source-reported |
| Total observations | not reported | source-reported |
| Frequency | not recorded (the id has no frequency suffix) | source-reported |
| Short-term test windows | not reported | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Standard protocol and known pitfalls

- Split (loader behavior): per series the last `pred_len * windows` steps are test, the preceding `pred_len` steps validation, the rest train. `windows` is `min(max(1, ceil(0.1 * shortest_series_length / pred_len)), 20)` computed from the shortest series (M4 ids use 1), so it can differ from the per-dataset window counts in the benchmark paper.
- Scaling: one StandardScaler per channel is fitted on the concatenated training regions of all series, not per series. Series shorter than `seq_len + pred_len` are skipped in train/val; test windows with too little context are dropped.
- Context length is not fixed by GIFT-Eval; pick `seq_len` per model and report it. Leaderboard metrics (MASE, CRPS) are scale-free and probabilistic; scores from this preset are comparable only when the same metric and windows are used.
- Leakage: the pretraining corpora of several foundation models overlap GIFT-Eval test data (source-reported). Use the GIFT-Eval pretrain split or a clean corpus when claiming zero-shot results.
- Multivariate: `features = "M"` forecasts all variates; `S` keeps the first variate only.
- The id has no frequency suffix, unlike other multi-frequency GIFT-Eval datasets (the paper lists Jena at 10T, H and D). Check that `dataset/gift_eval/jena_weather` exists after download; not verified locally.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `gift_eval`
- Config: [`configs/datasets/gift_eval/jena_weather.toml`](../../../../configs/datasets/gift_eval/jena_weather.toml)
- Local path: `./dataset/gift_eval`
- Dataset id: `jena_weather`
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

Inspect availability with `tsf data inspect --config configs/datasets/gift_eval/jena_weather.toml`; fetch
published files with `tsf data download gift_eval/jena_weather` when the preset is
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

- [`weather`](../../weather/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.

---
name: "gift_eval"
kind: "dataset-family"
summary: "GIFT-Eval is Salesforce's zero-shot-oriented benchmark for general time series forecasting; this repository ships 53 of its dataset-frequency series as short-term presets."
domain: "General / Energy, Finance, Healthcare, Nature, Sales, Transport, Web/CloudOps"
tags: ["gift-eval", "benchmark", "zero-shot", "foundation-model", "multi-domain", "multi-frequency", "salesforce"]
source: "Salesforce AI Research"
source_url: "https://huggingface.co/datasets/Salesforce/GiftEval"
citation: "GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation (Aksu et al., 2024)"
citation_url: "https://arxiv.org/abs/2410.10393"
license: "Apache-2.0 (benchmark packaging; underlying datasets keep their own licenses)"
redistribution: "conditional"
frequency: "mixed (10s, 5min, 10min, 15min, hourly, daily, weekly, monthly, quarterly, yearly)"
protocol: "Short, medium, long terms per dataset; last 10% of each series is test with rolling non-overlapping windows (at most 20); context length chosen by the model; this repository ships the short term"
stats_basis: "source-reported"
related: ["etth1", "ettm1", "electricity", "solar", "weather", "pems_bay", "metr_la"]
---

# gift_eval

## Overview

GIFT-Eval (General Time Series Forecasting Model Evaluation) was built by Salesforce AI Research to compare time series foundation models and classical or deep baselines on one broad, zero-shot-oriented suite. The benchmark holds 23 datasets across 7 domains and 10 frequencies (144,000 series, 177 million observations; 97 dataset, frequency, and term configurations), mixing 15 univariate and 8 multivariate datasets (source-reported). TSFLab ships 53 of the dataset-frequency series as presets named `gift_eval/<id>`, each at the short-term horizon.

Use a family member when you need breadth (many domains, frequencies, series counts, and horizons) rather than the single long-horizon multivariate protocols of the LTSF presets (`etth1`, `electricity`, `weather`, ...). Each series card links back here.

## Provenance and license

- Authors: Taha Aksu, Gerald Woo, Juncheng Liu, Xu Liu, Chenghao Liu, Silvio Savarese, Caiming Xiong, Doyen Sahoo (Salesforce AI Research).
- Paper: https://arxiv.org/abs/2410.10393. Data: https://huggingface.co/datasets/Salesforce/GiftEval. Code: https://github.com/SalesforceAIResearch/gift-eval. Leaderboard: https://huggingface.co/spaces/Salesforce/GIFT-Eval.
- License of the benchmark packaging: `apache-2.0` (Hugging Face dataset card; the repository LICENSE is Apache License 2.0, Copyright 2024 Salesforce, Inc.).
- Each underlying dataset keeps its own license. The member cards record it: CC-BY-4.0 for the Monash, M4, KDD Cup 2018, and UCI Electricity series; CC-BY-ND-4.0 for ETT; CDLA-Sharing-1.0 for BizITObs; and `unknown` for the Solar, LOOP Seattle, M-DENSE, SZ-Taxi, Bitbrains, Jena weather, Hierarchical Sales, and Restaurant sources. `redistribution: "unknown"` on a member means no explicit terms were found, not that republishing is allowed.
- TSFLab does not bundle these bytes. Download with `tsf data prepare --from gift`; it links `./dataset/gift_eval` to the download directory.

## Structure and statistics

All numbers are source-reported (GIFT-Eval paper, Table 13, and the repository's dataset properties); nothing is measured locally because the data are not bundled.

- Benchmark: 23 datasets, 144,000 series, 177 million observations, 7 domains, 10 frequencies, 97 configurations; 8 multivariate datasets (Jena Weather, ETT1/ETT2, BizITObs Application/Service/L2C, Bitbrains Fast Storage/Rnd) and 15 univariate ones.
- This repository: 53 presets, each one dataset-frequency pair at the short-term horizon. Series counts range from 1 (ETT, saugeenday, us_births) to 48,000 (M4 monthly); the per-series cards list counts, lengths, and horizons, and the generated table below lists the members.
- Short-term horizons follow frequency: 60 for 10s, 48 for 5min/10min/15min/hourly, 30 for daily, 8 for weekly, 12 for monthly, 8 for quarterly, 6 for yearly; M4 keeps its competition horizons (14 daily, 13 weekly, 18 monthly, 48 hourly, 8 quarterly, 6 yearly). Medium and long terms extend the short horizon by 10x and 15x where enough windows exist.

## Standard protocol and known pitfalls

- Official protocol: the last 10% of each series is test, scored with non-overlapping rolling windows of length equal to the horizon (at most 20 windows); the window before the test region serves as validation. Context length is chosen by the model.
- The loader in this repository derives the number of test windows from the shortest series of the dataset (`min(max(1, ceil(0.1 * min_len / pred_len)), 20)`, 1 for M4), which can differ from the per-dataset counts in the paper. Inspect `tsf catalog show gift_eval/<id>` and the loader before comparing against leaderboard numbers.
- The scaler is fitted on the training regions of all series together; z-scored losses weigh series by their original variance unlike the scale-free official metrics (MASE, CRPS on the leaderboard).
- Leakage: the pretraining corpora of TimesFM, Chronos, and Moirai partly overlap GIFT-Eval test data (source-reported). Zero-shot claims need the GIFT-Eval pretrain split or a verified clean corpus.
- `*_with_missing` series contain NaNs that the loader passes through; handle them in the model or the task.
- Multi-frequency datasets (electricity, ETT, solar, ...) repeat the same underlying data at several resolutions; do not treat them as independent datasets when aggregating scores.

<!-- dataset-card:canonical:start -->
## Members and loader

- Registry loader: `gift_eval`
- Data root: `./dataset/gift_eval`; each preset selects a series by `id`.
- Presets: 53

| Preset | Domain | Frequency | Horizon |
| --- | --- | --- | --- |
| [`gift_eval/LOOP_SEATTLE_5T`](LOOP_SEATTLE_5T/README.md) | Transport | 5-minute (5min) | 48 |
| [`gift_eval/LOOP_SEATTLE_D`](LOOP_SEATTLE_D/README.md) | Transport | daily (1d) | 30 |
| [`gift_eval/LOOP_SEATTLE_H`](LOOP_SEATTLE_H/README.md) | Transport | hourly (1h) | 48 |
| [`gift_eval/M_DENSE_D`](M_DENSE_D/README.md) | Transport | daily (1d) | 30 |
| [`gift_eval/M_DENSE_H`](M_DENSE_H/README.md) | Transport | hourly (1h) | 48 |
| [`gift_eval/SZ_TAXI_15T`](SZ_TAXI_15T/README.md) | Transport | 15-minute (15min) | 48 |
| [`gift_eval/SZ_TAXI_H`](SZ_TAXI_H/README.md) | Transport | hourly (1h) | 48 |
| [`gift_eval/bitbrains_fast_storage_5T`](bitbrains_fast_storage_5T/README.md) | Web/CloudOps | 5-minute (5min) | 48 |
| [`gift_eval/bitbrains_fast_storage_H`](bitbrains_fast_storage_H/README.md) | Web/CloudOps | hourly (1h) | 48 |
| [`gift_eval/bitbrains_rnd_5T`](bitbrains_rnd_5T/README.md) | Web/CloudOps | 5-minute (5min) | 48 |
| [`gift_eval/bitbrains_rnd_H`](bitbrains_rnd_H/README.md) | Web/CloudOps | hourly (1h) | 48 |
| [`gift_eval/bizitobs_application`](bizitobs_application/README.md) | Web/CloudOps | 10-second (10s) | 60 |
| [`gift_eval/bizitobs_l2c_5T`](bizitobs_l2c_5T/README.md) | Web/CloudOps | 5-minute (5min) | 48 |
| [`gift_eval/bizitobs_l2c_H`](bizitobs_l2c_H/README.md) | Web/CloudOps | hourly (1h) | 48 |
| [`gift_eval/bizitobs_service`](bizitobs_service/README.md) | Web/CloudOps | 10-second (10s) | 60 |
| [`gift_eval/car_parts_with_missing`](car_parts_with_missing/README.md) | Sales | monthly (1mo) | 12 |
| [`gift_eval/covid_deaths`](covid_deaths/README.md) | Healthcare | daily (1d) | 30 |
| [`gift_eval/electricity_15T`](electricity_15T/README.md) | Energy | 15-minute (15min) | 48 |
| [`gift_eval/electricity_D`](electricity_D/README.md) | Energy | daily (1d) | 30 |
| [`gift_eval/electricity_H`](electricity_H/README.md) | Energy | hourly (1h) | 48 |
| [`gift_eval/electricity_W`](electricity_W/README.md) | Energy | weekly (1w) | 8 |
| [`gift_eval/ett1_15T`](ett1_15T/README.md) | Energy | 15-minute (15min) | 48 |
| [`gift_eval/ett1_D`](ett1_D/README.md) | Energy | daily (1d) | 30 |
| [`gift_eval/ett1_H`](ett1_H/README.md) | Energy | hourly (1h) | 48 |
| [`gift_eval/ett1_W`](ett1_W/README.md) | Energy | weekly (1w) | 8 |
| [`gift_eval/ett2_15T`](ett2_15T/README.md) | Energy | 15-minute (15min) | 48 |
| [`gift_eval/ett2_D`](ett2_D/README.md) | Energy | daily (1d) | 30 |
| [`gift_eval/ett2_H`](ett2_H/README.md) | Energy | hourly (1h) | 48 |
| [`gift_eval/ett2_W`](ett2_W/README.md) | Energy | weekly (1w) | 8 |
| [`gift_eval/hierarchical_sales_D`](hierarchical_sales_D/README.md) | Sales | daily (1d) | 30 |
| [`gift_eval/hierarchical_sales_W`](hierarchical_sales_W/README.md) | Sales | weekly (1w) | 8 |
| [`gift_eval/hospital`](hospital/README.md) | Healthcare | monthly (1mo) | 12 |
| [`gift_eval/jena_weather`](jena_weather/README.md) | Nature | not recorded (the id has no frequency suffix) | 48 |
| [`gift_eval/kdd_cup_2018_with_missing_D`](kdd_cup_2018_with_missing_D/README.md) | Nature | daily (1d) | 30 |
| [`gift_eval/kdd_cup_2018_with_missing_H`](kdd_cup_2018_with_missing_H/README.md) | Nature | hourly (1h) | 48 |
| [`gift_eval/m4_daily`](m4_daily/README.md) | Finance / economics | daily (1d) | 14 |
| [`gift_eval/m4_hourly`](m4_hourly/README.md) | Finance / economics | hourly (1h) | 48 |
| [`gift_eval/m4_monthly`](m4_monthly/README.md) | Finance / economics | monthly (1mo) | 18 |
| [`gift_eval/m4_quarterly`](m4_quarterly/README.md) | Finance / economics | quarterly (1q) | 8 |
| [`gift_eval/m4_weekly`](m4_weekly/README.md) | Finance / economics | weekly (1w) | 13 |
| [`gift_eval/m4_yearly`](m4_yearly/README.md) | Finance / economics | yearly (1y) | 6 |
| [`gift_eval/restaurant`](restaurant/README.md) | Sales | daily (1d) | 30 |
| [`gift_eval/saugeenday_D`](saugeenday_D/README.md) | Nature | daily (1d) | 30 |
| [`gift_eval/saugeenday_M`](saugeenday_M/README.md) | Nature | monthly (1mo) | 12 |
| [`gift_eval/saugeenday_W`](saugeenday_W/README.md) | Nature | weekly (1w) | 8 |
| [`gift_eval/solar_10T`](solar_10T/README.md) | Energy | 10-minute (10min) | 48 |
| [`gift_eval/solar_D`](solar_D/README.md) | Energy | daily (1d) | 30 |
| [`gift_eval/solar_H`](solar_H/README.md) | Energy | hourly (1h) | 48 |
| [`gift_eval/solar_W`](solar_W/README.md) | Energy | weekly (1w) | 8 |
| [`gift_eval/temperature_rain_with_missing`](temperature_rain_with_missing/README.md) | Nature | daily (1d) | 30 |
| [`gift_eval/us_births_D`](us_births_D/README.md) | Healthcare | daily (1d) | 30 |
| [`gift_eval/us_births_M`](us_births_M/README.md) | Healthcare | monthly (1mo) | 12 |
| [`gift_eval/us_births_W`](us_births_W/README.md) | Healthcare | weekly (1w) | 8 |

## Composition constraints

Task modes: `time_series`. Each preset fixes `pred_len` and `features`; change them
only through an explicit task override. The family shares one loader, so
loader-level facts (splitting, scaling) are documented in the per-series
generated blocks.
<!-- dataset-card:canonical:end -->

## Related datasets

- LTSF counterparts: [`etth1`](../etth1/README.md), [`ettm1`](../ettm1/README.md), [`electricity`](../electricity/README.md), [`solar`](../solar/README.md), [`weather`](../weather/README.md).
- Traffic with spatial structure: [`pems_bay`](../pems_bay/README.md), [`metr_la`](../metr_la/README.md).

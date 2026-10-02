---
name: "ili"
kind: "dataset"
summary: "Weekly US influenza-like-illness surveillance counts and percentages from CDC FluView (seven channels), prepared for Autoformer; a very short, strongly seasonal LTSF benchmark."
domain: "Healthcare / epidemiology"
tags: ["ili", "influenza", "illness", "flu", "cdc", "fluview", "epidemiology", "weekly", "ltsf", "benchmark", "multivariate", "short-series"]
source: "US Centers for Disease Control and Prevention (FluView ILINet); packaged by Wu et al. (Autoformer)"
source_url: "https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html"
citation: "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting (Wu et al., NeurIPS 2021)"
citation_url: "https://arxiv.org/abs/2106.13008"
license: "US federal public domain; attribution to CDC and a no-endorsement disclaimer required (CDC Use of Agency Materials; raw source only, the Autoformer package states no data license)"
redistribution: "conditional"
frequency: "weekly (1w)"
time_span: "2002 to 2020 (file dates 2002-01-01 to 2020-06-30; Autoformer says 2002-2021)"
length: 966
channels: 7
channel_kind: "channels"
target: "OT (last column)"
missing_values: "none in the standard file"
protocol: "TSFLab: chronological 7:1:2 split, scaling fitted on the training split only; lookback 36/104, horizons 24/36/48/60"
literature_protocol: "LTSF: lookback 36, horizons 24/36/48/60, chronological 7:1:2, MSE/MAE on z-scored data"
seq_lens: [36, 104]
pred_lens: [24, 36, 48, 60]
split: "7:1:2"
stats_basis: "source-reported"
related: ["exchange", "fred_md", "nn5"]
config: "configs/datasets/ili.toml"
loader: "custom"
alias: "ili"
task_modes: ["time_series"]
---

# ili

## Overview

ILI (national_illness) is the weekly record of influenza-like-illness surveillance in the United States from the CDC's ILINet: weighted and unweighted ILI percentages, counts by age group, total ILI cases, and the number of reporting providers, with `OT` as the target. Autoformer popularized it for LTSF with horizons of 24 to 60 weeks. At 966 weekly steps it is the shortest standard LTSF set, so training windows are scarce and results are noisy.

## Provenance and license

- Source: US CDC FluView ILINet, https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html (the dashboard itself states no dataset license). CDC's Use of Agency Materials policy (https://www.cdc.gov/other/agencymaterials.html) says: "Most of the information on the CDC and ATSDR websites is not subject to copyright, is in the public domain, and may be freely used or reproduced without obtaining copyright permission", subject to attribution to CDC, a disclaimer that CDC does not endorse the use, and no change to the substantive content; some pages carry third-party or state/local material that may be copyrighted. Applying that to ILINet is an inference from CDC's general policy, not a dataset license.
- Packaging: Wu et al. 2021 (Autoformer), https://arxiv.org/abs/2106.13008; repository https://github.com/thuml/Autoformer is MIT (code), and the data files state no separate terms. Redistribution of the packaged `national_illness.csv` is therefore `conditional` on CDC attribution and disclaimer.
- This preset points to `dataset/illness/national_illness.csv`, which is not shipped.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows (standard file) | 966 weekly steps | source-reported (TFB data file) |
| Channels | 7: % WEIGHTED ILI, %UNWEIGHTED ILI, AGE 0-4, AGE 5-24, ILITOTAL, NUM. OF PROVIDERS, OT | source-reported |
| Span | file dates 2002-01-01 to 2020-06-30; Autoformer text says 2002-2021 | source-reported (conflicting) |
| Missing values | none in the standard file | source-reported |

The repository neither ships nor pins this file (`dataset/` is local and the Hub has no published copy), so the numbers below are those of the standard public distribution and a different copy may differ. Run `tsf data inspect --config configs/datasets/ili.toml` on your copy before relying on them.

## Standard protocol and known pitfalls

- **Protocol.** Lookback 36 (TFB also searches 104), horizons 24/36/48/60, split 7:1:2 (about 193 weekly test rows), so only a few dozen test windows exist at the longest horizon.
- **Short test set.** Metrics have high variance and depend on exact window handling, including `drop_last`; report seeds and window counts.
- **Seasonality.** A strong annual flu-season peak means only a few seasons fall in the test split. If the file's dates are real, the test split (about 2017 to mid-2020) includes the first COVID-19 weeks while the 2009 H1N1 outbreak sits in training; the date column is not verified.
- **Date column.** The standard file's dates are nominal; do not use them as a calendar for external covariates without checking.
- **`drop_last`.** Loaders keep the last partial batch for every split.
- **Target column.** `features = "S"` forecasts `OT`; `"MS"` forecasts the last column.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `custom`
- Config: [`configs/datasets/ili.toml`](../../../configs/datasets/ili.toml)
- Local path: `./dataset/illness/national_illness.csv`
- Dataset id: `(not applicable)`
- Track: `standard`

## Input and output contract

Each item provides history/target windows and timestamp marks; after batching, values use `[batch, time, channels]`.

Sequence length, label length, feature mode, and batch size are supplied by the
experiment task unless explicitly overridden below.

## Dataset parameters

```json
{
  "scale": true,
  "split_ratio": [
    0.7,
    0.1,
    0.2
  ],
  "target": "OT"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf data inspect --config configs/datasets/ili.toml`; fetch
published files with `tsf data download ili` when the preset is
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

- [`exchange`](../exchange/README.md): other small non-seasonal LTSF set
- [`fred_md`](../fred_md/README.md): other short low-frequency macro set
- [`nn5`](../nn5/README.md): short daily ATM set

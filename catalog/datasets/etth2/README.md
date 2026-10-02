---
name: "etth2"
kind: "dataset"
summary: "ETTh2: hourly oil temperature (OT) and six power-load features of one electricity transformer in China (station 2 of ETT-small), the standard long-horizon forecasting benchmark from Informer."
domain: "Energy / electricity transformer"
tags: ["ett", "etth2", "transformer", "oil-temperature", "power-load", "energy", "ltsf", "benchmark", "multivariate", "hourly", "informer"]
source: "Zhou et al. (Beihang University) with Beijing Guowang Fuda Science & Technology Development Co."
source_url: "https://github.com/zhouhaoyi/ETDataset"
citation: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (Zhou et al., AAAI 2021)"
citation_url: "https://arxiv.org/abs/2012.07436"
license: "CC-BY-ND-4.0"
redistribution: "conditional"
frequency: "hourly (1h)"
time_span: "2016-07-01 to 2018-02-20 (first 14,400 rows used; the CSV runs to 2018-06-26)"
length: 14400
channels: 7
channel_kind: "channels"
target: "OT"
missing_values: "none (measured: no empty cells, no duplicate or missing timestamps)"
protocol: "LTSF: lookback 96 (336 and 512 also searched), horizons 96/192/336/720, chronological 6:2:2 (12/4/4 months), MSE/MAE on z-scored data"
seq_lens: [96, 336, 512]
pred_lens: [96, 192, 336, 720]
split: "6:2:2"
stats_basis: "mixed"
related: ["ettm2", "etth1", "gift_eval/ett2_H"]
config: "configs/datasets/etth2.toml"
loader: "ETTh2"
alias: "etth2"
task_modes: ["time_series"]
---

# etth2

## Overview

ETT-small (Electricity Transformer Temperature) records two years of data from two transformer stations in China. ETTh2 is the hourly series of station 2: six power-load features (HUFL/HULL high, MUFL/MULL middle, LUFL/LULL low useful and useless load) and the oil temperature `OT`, the quantity the dataset was built to forecast because oil temperature indicates transformer overload risk. It was introduced with Informer (AAAI 2021) and became the default long-sequence benchmark, so almost every LTSF paper reports it (ettm2 is the same station at the other resolution; etth1 is the sibling series).

## Provenance and license

- Producer: Haoyi Zhou et al. (Beihang University) with Beijing Guowang Fuda Science & Technology Development Co.; repository https://github.com/zhouhaoyi/ETDataset.
- Cite: Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (Zhou et al., AAAI 2021), https://arxiv.org/abs/2012.07436. The upstream README asks users to cite Informer.
- License: Creative Commons Attribution-NoDerivatives 4.0 International (the repository's LICENSE file). Verbatim copies with attribution are allowed; a modified, re-split, or re-packaged copy is a derivative and is not covered, so publish only unmodified files. The Informer code repository is Apache-2.0 (separate from the data).
- Hub: `tsf dataset download --list` shows whether this preset is published; check the license before running `tsf dataset publish`.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| CSV rows | 17,420 | measured |
| Rows used by the loader | 14,400 (to 2018-02-20 23:00:00) | measured (loader code) |
| Channels | 7: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT | measured |
| Time span (CSV) | 2016-07-01 00:00:00 to 2018-06-26 19:00:00 | measured |
| Step | hourly (all 17,419 gaps equal; no duplicates) | measured |
| Missing values | 0 | measured |
| Exact zeros | 9.79% of all values | measured |
| OT mean / std (used rows) | 26.231 / 12.327 | measured |
| OT range (CSV) | -2.65 to 58.88 | measured |
| Split rows (6:2:2) | train 8,640, validation 2,880, test 2,880 | measured (loader code) |
| Provenance statement | two stations in two regions of one Chinese province; roughly two years | source-reported (upstream README, Informer paper) |

Measured on `dataset/ETT-small/ETTh2.csv` (read-only).

## Standard protocol and known pitfalls

- **Truncation.** The loader reads only the first 14,400 rows (20 months), so the last 3,020 rows of the 17,420-row CSV are never used (measured). The 20 months split chronologically as 12/4/4 months, which the ratio `[0.6, 0.2, 0.2]` reproduces exactly: train rows 0-8,639, validation to row 11,520, test to row 14,400.
- **Split convention.** ETT uses 6:2:2; most other LTSF sets use 7:1:2. Do not copy the ratio from another preset, and do not compare numbers across the two conventions.
- **Window overlap.** Validation and test splits start `seq_len` rows early so the first target window has full context. This reuses earlier rows only as inputs; scaling statistics are fitted on the training rows alone (`scale = true`).
- **`drop_last`.** The loaders keep the final partial batch (`drop_last=False`) for every split, so test metrics cover every window. Reference code that drops the last test batch reports slightly different numbers, more so at large batch sizes.
- **Target.** `features = "S"` forecasts `OT`; `"MS"` forecasts the last column, which is also `OT` here; `"M"` forecasts all seven channels.
- **Frequency naming.** `ETTm*` is 15-minute data, not one-minute data, although the upstream README text says "every minute" (the CSV is verified 15-minute).
- **Distribution shift.** ETTh2 has 9.8% exact zeros across all values (measured); the ETTh2/ETTm2 loads are zero-heavy compared with station 1 (about 1.0%) and shift more between train and test.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `ETTh2`
- Config: [`configs/datasets/etth2.toml`](../../../configs/datasets/etth2.toml)
- Local path: `./dataset/ETT-small/ETTh2.csv`
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
    0.6,
    0.2,
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

Inspect availability with `tsf dataset inspect --config configs/datasets/etth2.toml`; fetch
published files with `tsf dataset download etth2` when the preset is
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

- [`ettm2`](../ettm2/README.md): same station at the other resolution
- [`etth1`](../etth1/README.md): other station, same resolution
- [`gift_eval/ett2_H`](../gift_eval/ett2_H/README.md): GIFT-Eval resampling of the full series, different protocol

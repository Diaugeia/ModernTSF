---
name: "ettm1"
kind: "dataset"
summary: "ETTm1: 15-minute oil temperature (OT) and six power-load features of one electricity transformer in China (station 1 of ETT-small), the standard long-horizon forecasting benchmark from Informer."
domain: "Energy / electricity transformer"
tags: ["ett", "ettm1", "transformer", "oil-temperature", "power-load", "energy", "ltsf", "benchmark", "multivariate", "15min", "informer"]
source: "Zhou et al. (Beihang University) with Beijing Guowang Fuda Science & Technology Development Co."
source_url: "https://github.com/zhouhaoyi/ETDataset"
citation: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (Zhou et al., AAAI 2021)"
citation_url: "https://arxiv.org/abs/2012.07436"
license: "CC-BY-ND-4.0"
redistribution: "conditional"
frequency: "15-minute (15min)"
time_span: "2016-07-01 to 2018-02-20 (first 57,600 rows used; the CSV runs to 2018-06-26)"
length: 57600
channels: 7
channel_kind: "channels"
target: "OT"
missing_values: "none (measured: no empty cells, no duplicate or missing timestamps)"
protocol: "TSFLab: chronological 6:2:2 split, scaling fitted on the training split only; lookback 96/336/512, horizons 96/192/336/720"
literature_protocol: "LTSF: lookback 96 (336 and 512 also searched), horizons 96/192/336/720, chronological 6:2:2 (12/4/4 months), MSE/MAE on z-scored data"
seq_lens: [96, 336, 512]
pred_lens: [96, 192, 336, 720]
split: "6:2:2"
stats_basis: "mixed"
related: ["etth1", "ettm2", "gift_eval/ett1_H"]
config: "configs/datasets/ettm1.toml"
loader: "ETTm1"
alias: "ettm1"
task_modes: ["time_series"]
---

# ettm1

## Overview

ETT-small (Electricity Transformer Temperature) records two years of data from two transformer stations in China. ETTm1 is the 15-minute series of station 1: six power-load features (HUFL/HULL high, MUFL/MULL middle, LUFL/LULL low useful and useless load) and the oil temperature `OT`, the quantity the dataset was built to forecast because oil temperature indicates transformer overload risk. It was introduced with Informer (AAAI 2021) and became the default long-sequence benchmark, so almost every LTSF paper reports it (etth1 is the same station at the other resolution; ettm2 is the sibling series).

## Provenance and license

- Producer: Haoyi Zhou et al. (Beihang University) with Beijing Guowang Fuda Science & Technology Development Co.; repository https://github.com/zhouhaoyi/ETDataset.
- Cite: Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (Zhou et al., AAAI 2021), https://arxiv.org/abs/2012.07436. The upstream README asks users to cite Informer.
- License: Creative Commons Attribution-NoDerivatives 4.0 International (the repository's LICENSE file). Verbatim copies with attribution are allowed; a modified, re-split, or re-packaged copy is a derivative and is not covered, so publish only unmodified files. The Informer code repository is Apache-2.0 (separate from the data).
- Hub: `tsf dataset download --list` shows whether this preset is published; check the license before running `tsf dataset publish`.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| CSV rows | 69,680 | measured |
| Rows used by the loader | 57,600 (to 2018-02-20 23:45:00) | measured (loader code) |
| Channels | 7: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT | measured |
| Time span (CSV) | 2016-07-01 00:00:00 to 2018-06-26 19:45:00 | measured |
| Step | 15-minute (all 69,679 gaps equal; no duplicates) | measured |
| Missing values | 0 | measured |
| Exact zeros | 1.00% of all values | measured |
| OT mean / std (used rows) | 14.358 / 8.967 | measured |
| OT range (CSV) | -4.22 to 46.01 | measured |
| Split rows (6:2:2) | train 34,560, validation 11,520, test 11,520 | measured (loader code) |
| Provenance statement | two stations in two regions of one Chinese province; roughly two years | source-reported (upstream README, Informer paper) |

Measured on `dataset/ETT-small/ETTm1.csv` (read-only).

## Standard protocol and known pitfalls

- **Truncation.** The loader reads only the first 57,600 rows (20 months), so the last 12,080 rows of the 69,680-row CSV are never used (measured). The 20 months split chronologically as 12/4/4 months, which the ratio `[0.6, 0.2, 0.2]` reproduces exactly: train rows 0-34,559, validation to row 46,080, test to row 57,600.
- **Split.** TSFLab uses a chronological 6:2:2 split for this dataset. Results reported under other splits (7:1:2) are not directly comparable.
- **Window overlap.** Validation and test splits start `seq_len` rows early so the first target window has full context. This reuses earlier rows only as inputs; scaling statistics are fitted on the training rows alone (`scale = true`).
- **`drop_last`.** The loaders keep the final partial batch (`drop_last=False`) for every split, so test metrics cover every window. Reference code that drops the last test batch reports slightly different numbers, more so at large batch sizes.
- **Target.** `features = "S"` forecasts `OT`; `"MS"` forecasts the last column, which is also `OT` here; `"M"` forecasts all seven channels.
- **Frequency naming.** `ETTm*` is 15-minute data, not one-minute data, although the upstream README text says "every minute" (the CSV is verified 15-minute).
- **Distribution shift.** ETTm1 has 1.0% exact zeros across all values (measured); it is the cleanest ETT series and the usual first sanity benchmark.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `ETTm1`
- Config: [`configs/datasets/ettm1.toml`](../../../configs/datasets/ettm1.toml)
- Local path: `./dataset/ETT-small/ETTm1.csv`
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

Inspect availability with `tsf dataset inspect --config configs/datasets/ettm1.toml`; fetch
published files with `tsf dataset download ettm1` when the preset is
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

- [`etth1`](../etth1/README.md): same station at the other resolution
- [`ettm2`](../ettm2/README.md): other station, same resolution
- [`gift_eval/ett1_H`](../gift_eval/ett1_H/README.md): GIFT-Eval resampling of the full series, different protocol

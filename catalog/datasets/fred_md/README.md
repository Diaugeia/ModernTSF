---
name: "fred_md"
kind: "dataset"
summary: "FRED-MD monthly US macroeconomic indicators (McCracken and Ng, 2016) in the TFB packaging: 107 series, 728 months from 1959-01 to 2019-08."
domain: "Finance / macroeconomics"
tags: ["fred-md", "macroeconomic", "fred", "monthly", "economics", "us", "tfb", "multivariate", "short-series", "high-dimensional"]
source: "Michael W. McCracken and Serena Ng, Federal Reserve Bank of St. Louis; TFB packaging"
source_url: "https://www.stlouisfed.org/research/economists/mccracken/fred-databases"
citation: "FRED-MD: A Monthly Database for Macroeconomic Research (McCracken and Ng, Journal of Business & Economic Statistics 2016)"
citation_url: "https://doi.org/10.20955/wp.2015.012"
license: "unknown"
redistribution: "unknown"
frequency: "monthly (1mo)"
time_span: "1959-01 to 2019-08 (TFB file)"
length: 728
channels: 107
channel_kind: "series"
target: "OT (last series)"
missing_values: "the paper's database has 134 series; the TFB file keeps 107 (selection undocumented) with no NaN"
protocol: "TSFLab: chronological 7:1:2 split, scaling fitted on the training split only; lookback 36/104, horizons 24/36/48/60"
literature_protocol: "TFB multivariate: lookback 36 or 104, horizons 24/36/48/60, chronological 7:1:2"
seq_lens: [36, 104]
pred_lens: [24, 36, 48, 60]
split: "7:1:2"
stats_basis: "source-reported"
related: ["ili", "nn5", "exchange"]
config: "configs/datasets/fred_md.toml"
loader: "custom"
alias: "fred_md"
task_modes: ["time_series"]
---

# fred_md

## Overview

FRED-MD is a monthly database of US macroeconomic indicators (output, labor, housing, prices, money, interest rates, stock market) maintained at the St. Louis Fed from FRED data. TFB packaged 107 of the series for 728 months as a multivariate benchmark. The series are non-stationary, trend-dominated, and heterogeneous in scale, with only 728 observations.

## Provenance and license

- Source: FRED-MD by McCracken and Ng, https://www.stlouisfed.org/research/economists/mccracken/fred-databases; cite the 2016 JBES paper.
- License: checked the FRED-MD page (https://www.stlouisfed.org/research/economists/mccracken/fred-databases), which states no license, copyright or reuse terms for the CSV files, and FRED's legal terms (https://fred.stlouisfed.org/legal/), which say that before using series "owned by third parties for anything other than your own personal use, you must contact the data owner" and prohibit mirroring or scraping all of FRED; FRED-MD series are drawn from many original owners. No FRED-MD-specific grant exists, so `license` and `redistribution` stay `unknown`; do not publish copies.
- TFB packaging: https://arxiv.org/abs/2403.20150.
- This preset points to `dataset/FRED-MD/FRED-MD.csv`, which is not shipped.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 728 monthly steps | source-reported (TFB data file) |
| Channels | 107 (of 134 in the paper) | source-reported |
| Span | 1959-01 to 2019-08 | source-reported |
| Missing values | none in the TFB file | source-reported |

The repository neither ships nor pins this file (`dataset/` is local and the Hub has no published copy), so the numbers above are those of the standard public distribution and a different copy may differ. Run `tsf data inspect --config configs/datasets/fred_md.toml` on your copy before relying on them.

## Standard protocol and known pitfalls

- **Protocol.** Horizons 24/36/48/60 months with lookback 36 or 104 and split 7:1:2; the test split starts around mid-2007 (row 582 of 728, assuming the file starts in 1959-01) and therefore covers the 2008 financial crisis, with only about 146 test rows.
- **Revisions.** FRED-MD is revised monthly; a file downloaded later contains different vintages and a different series count than the TFB snapshot.
- **Transformations.** The standard FRED-MD recommends stationarity transformations per series; the preset forecasts levels as given in the file, which are trending.
- **`drop_last`.** Loaders keep the last partial batch for every split.
- **Target column.** `features = "S"` forecasts `OT`; `"MS"` forecasts the last column.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `custom`
- Config: [`configs/datasets/fred_md.toml`](../../../configs/datasets/fred_md.toml)
- Local path: `./dataset/FRED-MD/FRED-MD.csv`
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

Inspect availability with `tsf data inspect --config configs/datasets/fred_md.toml`; fetch
published files with `tsf data download fred_md` when the preset is
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

- [`ili`](../ili/README.md): other short low-frequency set
- [`nn5`](../nn5/README.md): other short set, daily
- [`exchange`](../exchange/README.md): other finance set

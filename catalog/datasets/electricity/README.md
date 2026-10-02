---
name: "electricity"
kind: "dataset"
summary: "Hourly electricity consumption of 321 Portuguese clients from 2012 to 2014 (the LSTNet/Autoformer preprocessing of the UCI load diagrams), a high-dimensional LTSF benchmark."
domain: "Energy / electricity consumption"
tags: ["electricity", "ecl", "load", "consumption", "energy", "ltsf", "benchmark", "multivariate", "hourly", "uci", "lstnet", "high-dimensional"]
source: "Artur Trindade (UCI ElectricityLoadDiagrams20112014); hourly 321-client preprocessing by Lai et al. (LSTNet)"
source_url: "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014"
citation: "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks (Lai et al., SIGIR 2018)"
citation_url: "https://arxiv.org/abs/1703.07015"
license: "CC-BY-4.0 (UCI raw data; the preprocessed LSTNet copy states no license)"
redistribution: "conditional"
frequency: "hourly (1h)"
time_span: "2012-01-01 00:00:01 to 2014-12-31 23:00:01"
length: 26304
channels: 321
channel_kind: "channels"
target: "all 321 clients (no OT column in this file)"
missing_values: "none; clients that started after 2012 carry zeros before their start; the 1-second timestamp offset is part of the file"
protocol: "LTSF: lookback 96 (336 and 512 also searched), horizons 96/192/336/720, chronological 7:1:2, MSE/MAE on z-scored data"
seq_lens: [96, 336, 512]
pred_lens: [96, 192, 336, 720]
split: "7:1:2"
stats_basis: "mixed"
related: ["traffic", "weather", "solar", "gift_eval/electricity_H"]
config: "configs/datasets/electricity.toml"
loader: "electricity"
alias: "electricity"
task_modes: ["time_series"]
---

# electricity

## Overview

Electricity (ECL) holds the electricity consumption of 321 clients of a Portuguese utility, aggregated to hourly values for 2012-2014. The raw UCI data (15-minute readings for 370 clients, 2011-2014) was reduced to 321 clients and hourly steps by LSTNet, and Informer, Autoformer, and Time-Series-Library carried that version forward. Each client is a channel, so it is the standard high-dimensional multivariate benchmark (321 channels) with strong daily and weekly seasonality and very different client scales.

## Provenance and license

- Raw data: Artur Trindade, UCI Machine Learning Repository, ElectricityLoadDiagrams20112014 (https://doi.org/10.24432/C58C86), license CC BY 4.0 ("sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given").
- Preprocessing: Lai et al. 2018 (LSTNet), https://github.com/laiguokun/multivariate-time-series-data. That repository carries no license file, so the preprocessed copy is derived from CC BY data and must credit Trindade.
- Cite LSTNet and the UCI dataset. UCI reports kW per 15 minutes while the LSTNet README says kWh, so the unit of this file is ambiguous; treat values as relative load.
- Redistribution: allowed with attribution for the raw data; prefer the UCI source when publishing.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 26,304 | measured |
| Channels | 321 (columns named 1 to 321; no `OT` column) | measured |
| Time span | 2012-01-01 00:00:01 to 2014-12-31 23:00:01, step 1 hour, no gaps or duplicates | measured |
| Missing values | 0 | measured |
| Value range | 0 to 764000, mean 2539, std 15028 | measured |
| Exact zeros | 1.09% of all values; clients mostly zero: 106, 108, 183, 299 | measured |
| Raw UCI data | 370 clients, 15-minute, 2011-2014 | source-reported (UCI) |

Measured on `dataset/electricity/electricity.csv` (read-only).

## Standard protocol and known pitfalls

- **Split.** The preset uses 7:1:2 (Autoformer/TFB convention); the original LSTNet paper used 6:2:2. Scaling statistics come from the training rows only.
- **Target column.** The preset sets `target = "OT"`, but this file has no `OT` column: `features = "S"` would raise a `KeyError`. Use `features = "M"` (all 321 clients) or `"MS"`, which forecasts the last column (client 321).
- **Scale heterogeneity.** Client magnitudes span several orders (maximum 764,000 versus mean 2,539), so per-channel z-scoring is essential and a few clients dominate unscaled errors.
- **Zero blocks.** Four clients are mostly zero and one client only starts after row 160 (measured); a zero is not always a measurement.
- **Timestamps** carry a one-second offset (`00:00:01`); calendar features are unaffected.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `electricity`
- Config: [`configs/datasets/electricity.toml`](../../../configs/datasets/electricity.toml)
- Local path: `./dataset/electricity/electricity.csv`
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

Inspect availability with `tsf dataset inspect --config configs/datasets/electricity.toml`; fetch
published files with `tsf dataset download electricity` when the preset is
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

- [`traffic`](../traffic/README.md): other 800+ channel hourly LTSF benchmark
- [`weather`](../weather/README.md): other standard 7:1:2 LTSF set
- [`solar`](../solar/README.md): other periodic energy benchmark
- [`gift_eval/electricity_H`](../gift_eval/electricity_H/README.md): GIFT-Eval electricity (370 clients), different protocol

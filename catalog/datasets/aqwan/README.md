---
name: "aqwan"
kind: "dataset"
summary: "Hourly air pollutants and weather at the Wanshouxigong station of the UCI Beijing Multi-Site Air Quality data (March 2013 to February 2017), 11 channels, as packaged by TFB."
domain: "Environment / air quality"
tags: ["air-quality", "beijing", "pm2.5", "pollution", "meteorology", "environment", "hourly", "tfb", "multivariate", "aqwan", "uci"]
source: "Beijing Municipal Environmental Monitoring Center and China Meteorological Administration data via UCI (donor Song Chen); TFB packaging by Qiu et al."
source_url: "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data"
citation: "Cautionary tales on air-quality improvement in Beijing (Zhang et al., Proceedings of the Royal Society A, 2017)"
citation_url: "https://doi.org/10.1098/rspa.2017.0457"
license: "CC-BY-4.0 (UCI page; the TFB repository's MIT license covers code, not data)"
redistribution: "conditional"
frequency: "hourly (1h)"
time_span: "2013-03-01 00:00 to 2017-02-28 23:00"
length: 35064
channels: 11
channel_kind: "channels"
target: "WSPM (wind speed, the last channel; the preset names it explicitly)"
missing_values: "the UCI original has NA cells (5,146 NA cells at Wanshouxigong); the TFB file has no NaN, so they were filled, and the categorical wind direction column was dropped"
protocol: "TSFLab: chronological 7:1:2 split, scaling fitted on the training split only; lookback 96/336/512, horizons 96/192/336/720"
literature_protocol: "TFB: lookback 96, 336 or 512, horizons 96/192/336/720, chronological 6:2:2; this preset's default split is 7:1:2"
seq_lens: [96, 336, 512]
pred_lens: [96, 192, 336, 720]
split: "7:1:2"
stats_basis: "source-reported"
related: ["aqshunyi", "beijing_air", "weather"]
config: "configs/datasets/aqwan.toml"
loader: "custom"
alias: "aqwan"
task_modes: ["time_series"]
---

# aqwan

## Overview

AQWan is the hourly record of one of 12 stations in the UCI Beijing Multi-Site Air Quality dataset, here the Wanshouxigong station: PM2.5, PM10, SO2, NO2, CO, O3, temperature, pressure, dew point, rain, and wind speed over four years. TFB (Qiu et al., 2024) packaged the two stations `AQShunyi` and `AQWan` as 11-channel multivariate benchmarks. Pollutant series are spiky and heavy-tailed, with strong dependence on weather and season.

## Provenance and license

- Source: UCI Machine Learning Repository, Beijing Multi-Site Air Quality Data (https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data), doi 10.24432/C5RK5G. Air data come from the Beijing Municipal Environmental Monitoring Center and weather from the China Meteorological Administration; the donor is Song Chen.
- License: Creative Commons Attribution 4.0 ("sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given"). The TFB package adds no separate data license.
- Cite Cautionary tales on air-quality improvement in Beijing (Zhang et al., Proceedings of the Royal Society A, 2017) and TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods (Qiu et al., PVLDB 2024).
- `AQWan` is the Wanshouxigong station, not Wanliu (verified from the TFB file's source name `PRSA_Data_Wanshouxigong_20130301-20170228.csv`).
- This preset points to `dataset/AQWan/AQWan.csv`, which is not shipped.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 35,064 hourly steps | source-reported (TFB data file) |
| Channels | 11: PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM | source-reported |
| Span | 2013-03-01 00:00 to 2017-02-28 23:00 | source-reported |
| Missing values | UCI original: 5,146 NA cells at this station (all columns); TFB file: none | source-reported |

The repository neither ships nor pins this file (`dataset/` is local and the Hub has no published copy), so the numbers above are those of the standard public distribution and a different copy may differ. Run `tsf data inspect --config configs/datasets/aqwan.toml` on your copy before relying on them.

## Standard protocol and known pitfalls

- **Split.** TSFLab uses a chronological 7:1:2 split for this dataset. Results reported under other splits (6:2:2) are not directly comparable.
- **Imputation.** The TFB file's NaN-free values were filled; the filling method is not documented, so errors at formerly missing hours are not real measurements.
- **Heavy tails.** PM2.5 and PM10 spike during haze episodes; z-scoring on the training split leaves outliers, and MSE is dominated by a few events.
- **Target column.** The preset sets `target = "WSPM"` (wind speed, the last channel), not a pollutant, so `features = "S"` and `"MS"` forecast wind speed; set `target` to a pollutant such as `PM2.5` if that is the goal. TFB has no `OT` convention: its reader keeps the channel names stored in the file (the original UCI names), so `OT` does not exist in the TFB files; this was derived from the TFB reader, not from a local copy. TFB ships these files in a long `date,data,cols` layout; the `custom` loader needs a wide CSV (`date` plus one column per channel), so pivot first. The loader raises if `target` is absent.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `custom`
- Config: [`configs/datasets/aqwan.toml`](../../../configs/datasets/aqwan.toml)
- Local path: `./dataset/AQWan/AQWan.csv`
- Dataset id: `(not applicable)`
- Track: `standard`

## Input and output contract

Each item provides history/target windows and timestamp marks `(year, month, day, weekday, hour, minute)` parsed from the file's date column; after batching, values use `[batch, time, channels]`.

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
  "target": "WSPM"
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf data inspect --config configs/datasets/aqwan.toml`. Not published and not downloadable with TSFLab: place the data file at the local path above (`./dataset/AQWan/AQWan.csv`) yourself; `tsf data prepare` has no converter for it. Reference this preset from an experiment configuration rather than duplicating its loader parameters.

## Composition constraints

Task modes: `time_series`. Match one of them to the model's declared task mode; the
config loader rejects other combinations. Change feature or scaling parameters
only after reading the loader. Paths are repository defaults and may need local
overrides; the card does not imply that the data is bundled.
<!-- dataset-card:canonical:end -->

## Related datasets

- [`aqshunyi`](../aqshunyi/README.md): other Beijing station, same packaging
- [`beijing_air`](../beijing_air/README.md): flat Beijing air-quality CSV preset with unrecorded origin
- [`weather`](../weather/README.md): other meteorological set

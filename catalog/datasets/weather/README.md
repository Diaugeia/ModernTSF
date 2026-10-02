---
name: "weather"
kind: "dataset"
summary: "21 meteorological indicators every 10 minutes during 2020 from the Max Planck Institute for Biogeochemistry weather station in Jena, as prepared for Autoformer."
domain: "Weather / meteorology"
tags: ["weather", "meteorology", "jena", "climate", "ltsf", "benchmark", "multivariate", "10min", "max-planck", "autoformer", "temperature", "pressure"]
source: "Max Planck Institute for Biogeochemistry, Jena (Beutenberg weather station)"
source_url: "https://www.bgc-jena.mpg.de/wetter/"
citation: "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting (Wu et al., NeurIPS 2021)"
citation_url: "https://arxiv.org/abs/2106.13008"
license: "CC-BY-4.0"
redistribution: "allowed"
frequency: "10-minute (10min)"
time_span: "2020-01-01 00:10 to 2020-12-31 22:40 (calendar year 2020)"
length: 52696
channels: 21
channel_kind: "channels"
target: "OT"
missing_values: "no NaN; the file holds -9999 sentinels in OT (50 rows), max. PAR (30) and wv (1), which the preset replaces with NaN and imputes causally (forward fill, back fill only at the series start) before scaling"
protocol: "TSFLab: -9999 sentinels imputed causally, chronological 7:1:2 split, scaling fitted on the training split only; lookback 96/336/512, horizons 96/192/336/720"
literature_protocol: "LTSF: lookback 96 (336 and 512 also searched), horizons 96/192/336/720, chronological 7:1:2, MSE/MAE on z-scored data"
seq_lens: [96, 336, 512]
pred_lens: [96, 192, 336, 720]
split: "7:1:2"
stats_basis: "measured"
related: ["ettm1", "electricity", "gift_eval/jena_weather"]
config: "configs/datasets/weather.toml"
loader: "weather"
alias: "weather"
task_modes: ["time_series"]
---

# weather

## Overview

Weather is the 2020 record of the roof weather station of the Max Planck Institute for Biogeochemistry in Jena, Germany: 21 indicators (air pressure, temperatures, humidity, vapor pressure, wind speed and direction, rain, radiation, and derived quantities) sampled every 10 minutes. Autoformer introduced this subset to the LTSF canon; it is the usual benchmark for many correlated channels with a smooth daily cycle and sparse intermittent channels such as rain.

## Provenance and license

- Producer: Max-Planck-Institut fuer Biogeochemie, Jena (station documentation by Olaf Kolle); data at https://www.bgc-jena.mpg.de/wetter/.
- License: the Data Download page (https://www.bgc-jena.mpg.de/wetter/weather_data.html) states "Terms of Use (as per Creative Commons CC-BY-4.0)"; the landing page itself states none. Redistribution and adaptation are allowed with attribution.
- Cite Autoformer for this subset and credit the Max Planck Institute for Biogeochemistry.
- Do not confuse it with Informer's own "Weather" set (NOAA local climatological data, 2010-2013), which is a different dataset.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 52,696 | measured |
| Channels | 21 (alphabetical order with `OT` second; the last column is `wv (m/s)`) | measured |
| Time span | 2020-01-01 00:10:00 to 2020-12-31 22:40:00; contiguous 10-minute grid, no gaps | measured |
| NaN values | 0 | measured |
| -9999 sentinels | OT: 50, max. PAR (�mol/m�/s): 30, wv (m/s): 1 | measured |
| OT range excluding sentinels | 305.5 to 524.2, mean 427.7 (looks like CO2 in ppm; meaning not verified) | measured |
| Exact zeros | 16.0% of all values (rain, raining, radiation at night) | measured |
| Year coverage | 2020; the series starts at 00:10 and ends at 22:40, so 8 slots of a full year grid are absent at the edges | measured |

Measured on `dataset/weather/weather.csv` (read-only).

## Standard protocol and known pitfalls

- **Split.** 7:1:2 chronological, scaling on training rows only. The test split is roughly mid-October to December 2020 and the training split is January to August, so seasonal shift between train and test is large.
- **Sentinels.** `-9999` marks missing readings in `OT` (50 rows), `max. PAR` (30), and `wv` (1). The preset sets `missing_sentinels = [-9999]`, so they become NaN and are filled causally before scaling (forward fill, then back fill only for a leading gap); all affected rows fall in the training split (rows 7,244 to 31,773 of 52,696, measured). Left in place they inflate the training standard deviation of `OT` from 18.2 to 384.0, of `max. PAR` from 573.5 to 643.4, and of `wv` from 1.67 to 52.1, with z-scores down to -192 (measured on the training split). Results therefore differ from papers that use the file as-is and from earlier TSFLab weather runs. Remove the parameter to reproduce the as-is protocol.
- **Target.** `"S"` and `"MS"` both forecast `OT`: the loader moves the named target to the last channel in `MS` mode, so the alphabetical column order of this file does not change the target.
- **Encoding.** Column names containing the micro and superscript characters were stored with replacement characters; do not select columns by those names.
- **Intermittent channels.** `rain (mm)` and `raining (s)` are mostly zero, so MSE is dominated by smooth channels while MAPE-style metrics are undefined.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `weather`
- Config: [`configs/datasets/weather.toml`](../../../configs/datasets/weather.toml)
- Local path: `./dataset/weather/weather.csv`
- Dataset id: `(not applicable)`
- Track: `standard`

## Input and output contract

Each item provides history/target windows and timestamp marks; after batching, values use `[batch, time, channels]`.

Sequence length, label length, feature mode, and batch size are supplied by the
experiment task unless explicitly overridden below.

## Dataset parameters

```json
{
  "missing_sentinels": [
    -9999
  ],
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

Inspect availability with `tsf data inspect --config configs/datasets/weather.toml`; fetch
published files with `tsf data download weather` when the preset is
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

- [`ettm1`](../ettm1/README.md): other sub-hourly LTSF set
- [`electricity`](../electricity/README.md): other 7:1:2 LTSF set
- [`gift_eval/jena_weather`](../gift_eval/jena_weather/README.md): GIFT-Eval packaging of the same station

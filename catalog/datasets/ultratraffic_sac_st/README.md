---
name: "ultratraffic_sac_st"
description: "Hourly total flow of all 801 Caltrans PeMS stations in District 3 (North Central, Sacramento) for 2023, stations as nodes plus two calendar covariates. Use for one-year, large-panel hourly flow forecasting with calendar covariates; not for graph models (no adjacency) or multi-year seasonality."
---

# ultratraffic_sac_st

## Overview

This preset loads 2023 hourly total flow (vehicles per hour, summed over lanes) for all 801 stations present in 2023 in Caltrans PeMS District 3 (North Central, Sacramento), from the local UltraTraffic parquet store (`dataset/ultratraffic/PEMS_SAC`, built from `UltraTraffic_CL.zip`): 8,760 hourly rows, stations as nodes plus time-of-day and day-of-week covariates. The store holds 2003-2023 (84 to 801 stations per year) and bootstraps the real-time track `traffic_pems_sac`.

## Protocol and pitfalls

- **One year only.** The preset loads 2023 alone: a 7:1:2 split puts training from January to mid-September (6,132 hours), validation to 19 October, and the last 1,752 hours (late October to December, with Thanksgiving and Christmas) in test, so the test period includes holiday regimes the training split lacks.
- **Scaling.** One scalar mean and standard deviation is computed on the training rows across all stations (not per station), so large stations dominate and per-station scale differences remain; the loader's `value_mean`/`value_std` are used for inversion.
- **Gap filling.** The loader linearly interpolates (both directions) and zero-fills the whole panel before splitting; for the static 2023 panel there are no gaps, but other variants or years could use future values to fill earlier gaps.
- **Zeros.** 4.38% of the 2023 values are exact zeros (measured); long zero runs at some stations are probably outages or closures rather than empty roads, not traced to the source.
- **No adjacency.** The archive carries no station coordinates, so graph models receive no adjacency (`adj_mx` is None).
- **Different from `traffic`.** This is hourly flow in vehicles per hour for 2023, not LTSF's 2015-2016 occupancy rate.
- **`drop_last`.** Loaders keep the last partial batch for every split.

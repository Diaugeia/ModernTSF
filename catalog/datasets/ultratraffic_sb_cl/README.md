---
name: "ultratraffic_sb_cl"
description: "Hourly total flow of the 51 Caltrans PeMS District 8 (San Bernardino / Riverside) stations first observed in 2023, as nodes plus two calendar covariates. Use for continual-learning studies of sensors that join a network; not for graph models (no adjacency) or as a full-district benchmark."
---

# ultratraffic_sb_cl

## Overview

This preset loads 2023 hourly total flow (vehicles per hour, summed over lanes) for the `cl_added` slice of Caltrans PeMS District 8 (San Bernardino / Riverside): only the 51 stations first observed in 2023, the continual-learning view of sensors that joined the network (the district has 1,105 stations in 2023). Data come from the local UltraTraffic parquet store (`dataset/ultratraffic/PEMS_SB`, built from `UltraTraffic_CL.zip`): 8,760 hourly rows, stations as nodes plus time-of-day and day-of-week covariates.

## Protocol and pitfalls

- **One year only.** The preset loads 2023 alone: a 7:1:2 split puts training from January to mid-September (6,132 hours), validation to 19 October, and the last 1,752 hours (late October to December, with Thanksgiving and Christmas) in test, so the test period includes holiday regimes the training split lacks.
- **Scaling.** One scalar mean and standard deviation is computed on the training rows across all stations (not per station), so large stations dominate and per-station scale differences remain; the loader's `value_mean`/`value_std` are used for inversion.
- **Gap filling.** The loader linearly interpolates (both directions) and zero-fills the whole panel before splitting; for the static 2023 panel there are no gaps, but other variants or years could use future values to fill earlier gaps.
- **Zeros.** The 51 added stations report exact zeros for 16% of their hours (8% for all District 8 stations, 4% for District 3), probably outages or closures rather than real empty roads; this was not traced to the source.
- **Continual-learning view.** `cl_added` contains only stations new in 2023; pair it with `ultratraffic_sb_st` (all stations) or the `cl_common` variant to study sensors joining the network.
- **No adjacency.** The archive carries no station coordinates, so graph models receive no adjacency (`adj_mx` is None).
- **Different from `traffic`.** This is hourly flow in vehicles per hour for 2023, not LTSF's 2015-2016 occupancy rate.
- **`drop_last`.** Loaders keep the last partial batch for every split.

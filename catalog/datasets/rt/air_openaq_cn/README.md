---
name: "rt/air_openaq_cn"
description: "Hourly PM2.5 from up to 200 government monitors in China via OpenAQ v3; frozen release of real-time track air_openaq_cn, seq_len 168, horizon 24. Use for hourly multi-station air-quality forecasting on noisy, irregular data; not for multi-year seasonality."
---

# rt/air_openaq_cn

## Overview

`rt_air_openaq_cn` serves the OpenAQ China PM2.5 real-time track as a static spatiotemporal dataset: hourly PM2.5 (micrograms per cubic metre) at up to 200 government monitors in China, aggregated by OpenAQ v3, with stations as nodes plus calendar covariates. History is a 365-day API backfill plus each weekly release. Use it to train and check models before submitting to the live track.

## Protocol and pitfalls

- **Sensor sets are fixed at bootstrap.** The channel set is frozen so rounds stay comparable; stations that start reporting later are not added, and stations that stop remain as forward-filled constants.
- **Reported values are noisy.** PM2.5 readings can be sparse, spiky, and include instrument noise; the real-time store keeps them as reported.
- **Time zone.** Stamps are naive UTC while the stations sit in UTC+8, so the local daily cycle is shifted by eight hours.
- **Frozen snapshot, not the live track.** No `version` is pinned, so the preset reads whatever the local panel store holds; pin a release or set `revision` for a reproducible study. The live track's rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation; a static split says nothing about data that arrive later.
- **Gap filling.** Unobserved cells are forward-filled; only a series' leading gap is back-filled from its first reading. The loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training; fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Loaders keep the last partial batch for every split.

---
name: "MoFo"
description: "Period-structured 2D patches with attention whose scores are biased by cycle distance through a learnable relaxation. Use for long-term forecasting of strongly periodic series with a known period; not for aperiodic data, cross-channel interaction, calendar inputs, or probabilistic output."
---

# MoFo

## Idea

- `period_structured_patches` reshapes the series into cycles by phase using the `periodic` length (default 24).
- Each future step is a learned query that attends only to the history values at the same phase across cycles (`PeriodModulatedAttention`).
- `RegulatedRelaxation` converts cycle distance into a learnable additive log-bias on the attention scores, so recent cycles can weigh more.
- Channels share weights (`cias` must be 1); `revin` without affine wraps the model and `projection` maps each query to a scalar.

## When to use

- Long-horizon forecasting of series with a stable, known period (for example daily cycles in hourly data): the model is built around period-aligned correlation.
- Tight memory or time budgets: each query attends only over one value per cycle; the paper reports high memory efficiency and fast training.
- Not for aperiodic or weakly seasonal data or a mis-specified period, when cross-channel interaction carries the signal, when calendar marks matter (ignored), or when quantiles are needed (point output only).

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count.
- `periodic`: the dataset's dominant period in steps (preset 24 for hourly data with a daily cycle). Keep `seq_len` a multiple of it: the history is left-padded to whole cycles while future phases are computed as `(seq_len + h) % periodic`, so a remainder shifts phase alignment.

Other hyperparameters: preset defaults in `configs/models/MoFo.toml`; tune generically.

## Differences

- Paper-driven local implementation of discrete period sampling, phase-restricted future queries, and the regulated relaxation function (Eq. 9, checked at its boundary values); MIT official code is reference-only, nothing copied.
- Only channel-independent sharing is supported (`cias = 0` raises); calendar marks are not used.

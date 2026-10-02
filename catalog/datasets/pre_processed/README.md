---
name: "pre_processed"
kind: "dataset"
summary: "A directory of pre-windowed train, validation, and test `.npz` files (x, y, x_mark, y_mark) prepared by `tsf dataset prepare`; the preset points at a user-created folder."
domain: "User-supplied / pre-windowed"
tags: ["pre-processed", "npz", "windowed", "custom", "user-data", "prepare", "time-series"]
source: "user-supplied (created by `tsf dataset prepare`)"
source_url: "n/a"
citation: "n/a"
citation_url: "n/a"
license: "unknown"
redistribution: "unknown"
frequency: "unknown (depends on the source data)"
time_span: "unknown (depends on the source data)"
length: "unknown (depends on the source data)"
channels: "unknown (depends on the source data)"
channel_kind: "channels"
target: "all prepared channels (window contents fixed at preparation time)"
missing_values: "unknown"
protocol: "Windows (seq_len, label_len, pred_len) and the split are fixed when the files are prepared; the default preparation split is 7:1:2"
seq_lens: [96]
pred_lens: [96]
split: "7:1:2"
stats_basis: "source-reported"
related: ["etth1"]
config: "configs/datasets/pre_processed.toml"
loader: "pre_processed"
alias: "pre_processed"
task_modes: ["time_series"]
---

# pre_processed

## Overview

`pre_processed` loads windows that were already cut and split by `tsf dataset prepare` (default folder `./dataset/my_dataset_npy`). Each of `train.npz`, `val.npz`, `test.npz` holds float32 arrays `x`, `y`, `x_mark`, `y_mark`, and optionally `scaler_mean`/`scaler_scale`. It is the escape hatch for data that the CSV loaders cannot express, so it has no source of its own.

## Provenance and license

- The data are whatever you prepared; the repository records no source, citation, or license for them. Record those facts in your own card when you add a named preset for a real dataset.

## Structure and statistics

No source-reported or measured statistics exist: the contents depend entirely on the prepared files, which the repository neither ships nor pins.

## Standard protocol and known pitfalls

- **Fixed windows.** `seq_len`, `label_len`, and `pred_len` were baked into the files; the experiment's task values must match them, because the loader returns the stored windows unchanged.
- **Scaling.** If `scaler_mean`/`scaler_scale` are stored, `inverse_transform` uses them; otherwise outputs stay in the stored scale.
- **Leakage.** Preparation determines whether windows overlap the split boundaries and which rows fit the scaler; check the preparation command, because the loader cannot.
- **`drop_last`.** Loaders keep the last partial batch for every split.

<!-- dataset-card:canonical:start -->
## Loader and files

- Registry loader: `pre_processed`
- Config: [`configs/datasets/pre_processed.toml`](../../../configs/datasets/pre_processed.toml)
- Local path: `./dataset/my_dataset_npy`
- Dataset id: `(not applicable)`
- Track: `standard`

## Input and output contract

Each item provides history/target windows and timestamp marks; after batching, values use `[batch, time, channels]`.

Sequence length, label length, feature mode, and batch size are supplied by the
experiment task unless explicitly overridden below.

## Dataset parameters

```json
{}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/pre_processed.toml`; fetch
published files with `tsf dataset download pre_processed` when the preset is
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

- [`etth1`](../etth1/README.md): CSV preset that applies the same split logic inside the loader

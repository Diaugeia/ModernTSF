---
name: "ultratraffic_la_ts"
kind: "dataset"
config: "configs/datasets/ultratraffic_la_ts.toml"
loader: "ultratraffic_ts"
alias: "ultratraffic_la_ts"
task_modes: ["time_series"]
summary: "Time-series forecasting preset loaded by `ultratraffic_ts`."
---

# ultratraffic_la_ts

## Overview

Time-series forecasting preset loaded by `ultratraffic_ts`. This card describes the repository preset and runtime contract; it
does not add an external-source provenance claim that is absent from the configuration.

## Loader and files

- Registry loader: `ultratraffic_ts`
- Config: [`configs/datasets/ultratraffic_la_ts.toml`](../../../configs/datasets/ultratraffic_la_ts.toml)
- Local path: `./dataset/ultratraffic`
- Dataset id: `(not applicable)`
- Track: `standard`

## Input and output contract

Each item provides history/target windows and timestamp marks; after batching, values use `[batch, time, channels]`.

Sequence length, label length, feature mode, and batch size are supplied by the
experiment task unless explicitly overridden below.

## Dataset parameters

```json
{
  "region": "PEMS_LA",
  "scale": true,
  "split_ratio": [
    0.7,
    0.1,
    0.2
  ],
  "variant": "static",
  "years": [
    2023
  ]
}
```

## Task overrides

```json
{}
```

## Preparation and use

Inspect availability with `tsf dataset inspect --config configs/datasets/ultratraffic_la_ts.toml` and
prepare/download data with the loader-specific dataset command when required.
Reference this preset from an experiment configuration rather than duplicating
its loader parameters.

## Composition constraints

Choose one of `time_series` and match it to the model's declared task
mode. Inspect the loader
before changing feature or scaling parameters. Paths are repository defaults and
may need local overrides; the card does not imply that the data is bundled.

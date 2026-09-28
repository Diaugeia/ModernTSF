---
name: "ultratraffic_sb_st"
kind: "dataset"
config: "configs/datasets/ultratraffic_sb_st.toml"
loader: "ultratraffic_st"
alias: "ultratraffic_sb_st"
task_modes: ["spatiotemporal", "covariate"]
summary: "Time-series forecasting preset loaded by `ultratraffic_st`."
---

# ultratraffic_sb_st

## Overview

Time-series forecasting preset loaded by `ultratraffic_st`. This card describes the repository preset and runtime contract; it
does not add an external-source provenance claim that is absent from the configuration.

## Loader and files

- Registry loader: `ultratraffic_st`
- Config: [`configs/datasets/ultratraffic_sb_st.toml`](../../../configs/datasets/ultratraffic_sb_st.toml)
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
  "region": "PEMS_SB",
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

Inspect availability with `tsf dataset inspect --config configs/datasets/ultratraffic_sb_st.toml` and
prepare/download data with the loader-specific dataset command when required.
Reference this preset from an experiment configuration rather than duplicating
its loader parameters.

## Composition constraints

Choose one of `spatiotemporal, covariate` and match it to the model's declared task
mode. Inspect the loader
before changing feature or scaling parameters. Paths are repository defaults and
may need local overrides; the card does not imply that the data is bundled.

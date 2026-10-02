---
name: "Composed"
summary: "Composed is a recombination model: its parameters are a slot assignment (normalization, decomposition, temporal, channel, head) executed through shared slot adapters that wrap cataloged components, so an autoresearch agent can run a validated composition without writing model code."
paper: "https://github.com/Diaugeia/ModernTSF/"
paper_title: "Composed: slot-assignment recombination of cataloged components"
venue: "unpublished"
year: 2026
tagline: "Six-slot recombination model: pick normalization, decomposition, temporal, channel, head from cataloged components."
tags: ["hybrid", "linear", "transformer", "ssm", "cnn", "mlp", "recombination", "autoresearch"]
composition: ["normalization=component:revin+component:last_value_center", "decomposition=component:series_decomposition", "temporal=component:channel_wise_linear+component:tst_transformer+component:mamba+component:gated_dilated_conv+component:mixer_block", "channel=local:slot-selected", "head=component:flatten_forecast_head", "loss=loss:mse+loss:mae"]
---
# Composed

## Key ideas

- Parameters are the slot assignment: `normalization`, `decomposition`, `temporal`, `channel`, `head`; the loss stays in `training.loss`.
- Adapters in `tsflab.models._slots` wrap each cataloged component with its documented interface; invalid combinations fail validation with the reason (for example `mixer_block` needs `channel = "mixing"`).
- Point output only (`flatten_forecast_head`); quantile and Gaussian heads need a dedicated entry made with `tsf component compose --register`.
- `tsf component compose spec.toml --write-config run.toml` turns a validated spec into a runnable config for this model.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://github.com/Diaugeia/ModernTSF/); title: Composed: slot-assignment recombination of cataloged components; venue/year: unpublished / 2026
- codebase: not available

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/Composed.toml`](../../../../configs/models/Composed.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

- [`channel_wise_linear`](../_components/channel_wise_linear/README.md)
- [`flatten_forecast_head`](../_components/flatten_forecast_head/README.md)
- [`gated_dilated_conv`](../_components/gated_dilated_conv/README.md)
- [`last_value_center`](../_components/last_value_center/README.md)
- [`mamba`](../_components/mamba/README.md)
- [`mixer_block`](../_components/mixer_block/README.md)
- [`positional_encoding`](../_components/positional_encoding/README.md)
- [`revin`](../_components/revin/README.md)
- [`series_decomposition`](../_components/series_decomposition/README.md)
- [`tst_transformer`](../_components/tst_transformer/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `normalization='revin'`, `decomposition='none'`, `temporal='linear'`, `channel='independent'`, `head='flatten_forecast_head'`
<!-- model-card:canonical:end -->

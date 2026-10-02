---
name: "Composed"
summary: "Recombination model whose parameters are a slot assignment (normalization, decomposition, temporal, channel, head) executed by shared slot adapters over cataloged components, so a validated composition runs without new model code. The loss is the sixth slot and lives in the run config."
paper: ""
paper_title: "Composed: slot-assignment recombination of cataloged components (TSFLab)"
venue: "TSFLab"
year: 2026
tagline: "Six-slot recombination: pick normalization, decomposition, temporal, channel and head from cataloged components."
tags: ["hybrid", "recombination", "autoresearch", "slot-assignment"]
composition: ["normalization=component:revin+component:last_value_center", "decomposition=component:series_decomposition", "temporal=component:channel_wise_linear+component:tst_transformer+component:mamba+component:gated_dilated_conv+component:mixer_block", "channel=local:slot-selected", "head=component:flatten_forecast_head", "loss=loss:mse+loss:mae"]
---
# Composed

## Key ideas

- Parameters are the slot assignment `normalization`, `decomposition`, `temporal`, `channel`, `head`; the sixth slot, the loss, stays in `training.loss`. The `composition` field lists the selectable options per slot (alternatives joined by `+`), not options used together.
- `tsflab.models._slots.registry` is the executable option table: `OPTIONS`, `canonical` (aliases such as `component:revin`) and `check_assignment`, which returns every reason a combination cannot run; `ModelParameterConfig` rejects those combinations at config validation (for example `mixer_block` needs `channel = "mixing"`, and `channel = "mixing"` needs `mixer_block`).
- `tsflab.models._slots.adapters.Pipeline` runs `normalization.norm`, then `decomposition.split` (one branch, or residual and trend for `series_decomposition`), one temporal encoder per branch producing `z [B, C, D, P]`, concatenation of the branches along `P`, the `flatten_forecast_head`, and `normalization.denorm`. `temporal = "linear"` has no encoder: the head's linear map over the flattened history is the temporal model.
- `channel` sets weight sharing: `independent` shares weights across channels, `individual` gives per-channel weights in the head and in `channel_wise_linear`, `mixing` is only provided by `mixer_block`'s feature MLP.
- `tsf model compose spec.toml` validates a recombination spec; `--write-config run.toml --dataset <name>` writes a runnable config for this model, and `--register NAME` scaffolds a dedicated catalog entry (needed for quantile or Gaussian heads).

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: Composed: slot-assignment recombination of cataloged components (TSFLab); venue/year: TSFLab / 2026
- codebase: not available

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/Composed.toml`](../../../../configs/models/Composed.toml).

## Differences

No paper and no official code: this is a TSFLab-originated mechanism, not a reproduction, so the card carries no `code`, `revision` or `license`. Each slot option keeps the interface documented in its component card; the differences of those components from their papers are recorded there. Inputs are `[B, seq_len, enc_in]` and the output is `[B, pred_len, enc_in]`; marks and decoder inputs are ignored, and with `features = "MS"` only the last channel is returned.

- **Executable options.** `normalization`: `none`, `revin` (non-affine), `last_value_center`. `decomposition`: `none`, `series_decomposition` (`kernel_size`, odd). `temporal`: `linear`, `channel_wise_linear`, `tst_transformer` (patched, sinusoidal positions, feed-forward width `2 * hidden`), `mamba` (patched, pre-norm residual blocks with `d_state=8`, `d_conv=4`, `expand=2`), `gated_dilated_conv` (causal gated convolutions, kernel 3, dilations `2**i`, residual), `mixer_block`. `channel`: `independent`, `individual`, `mixing`. `head`: `flatten_forecast_head`.
- **Declared but not executable.** The `decomposition = "wavelet"` option is in the schema but the registry marks it unsupported (the component has no uniform `[B, L, C]` branch interface), so it always fails validation. `quantile_head` and `gaussian_parameter_head` exist in the slot registry for generated dedicated models but `Composed` rejects them: it declares point output only.
- **Constraints.** `kernel_size` must be odd, `hidden` must be divisible by `n_heads` (even when the temporal option is not a transformer), and `patch_len` must not exceed `seq_len` for the patched options. `hidden`, `n_layers`, `n_heads`, `patch_len`, `stride`, `kernel_size` and `dropout` apply only to the options that use them. The schema defaults are `normalization = "none"` and `temporal = "linear"`; the preset sets `normalization = "revin"`.
- **Loss.** Any point loss from `training.loss` is valid; the head has no loss of its own and the model defines no `training_objective` or auxiliary loss.
- **Smoke.** The spec declares no smoke config; `tsf model compose spec.toml --write-config run.toml --dataset <name> --smoke` writes a one-epoch CPU config.
- **Evidence.** Slot validation and adapter tests are in `tests/test_composed.py`.

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

## Source and verification

No paper and no official code: this is a TSFLab-originated mechanism, not a reproduction, so the card carries no `code`, `revision` or `license`. Each slot option keeps the interface documented in its component card; the differences of those components from their papers are recorded there. Inputs are `[B, seq_len, enc_in]` and the output is `[B, pred_len, enc_in]`; marks and decoder inputs are ignored, and with `features = "MS"` only the last channel is returned.

- **Executable options.** `normalization`: `none`, `revin` (non-affine), `last_value_center`. `decomposition`: `none`, `series_decomposition` (`kernel_size`, odd). `temporal`: `linear`, `channel_wise_linear`, `tst_transformer` (patched, sinusoidal positions, feed-forward width `2 * hidden`), `mamba` (patched, pre-norm residual blocks with `d_state=8`, `d_conv=4`, `expand=2`), `gated_dilated_conv` (causal gated convolutions, kernel 3, dilations `2**i`, residual), `mixer_block`. `channel`: `independent`, `individual`, `mixing`. `head`: `flatten_forecast_head`.
- **Declared but not executable.** The `decomposition = "wavelet"` option is in the schema but the registry marks it unsupported (the component has no uniform `[B, L, C]` branch interface), so it always fails validation. `quantile_head` and `gaussian_parameter_head` exist in the slot registry for generated dedicated models but `Composed` rejects them: it declares point output only.
- **Constraints.** `kernel_size` must be odd, `hidden` must be divisible by `n_heads` (even when the temporal option is not a transformer), and `patch_len` must not exceed `seq_len` for the patched options. `hidden`, `n_layers`, `n_heads`, `patch_len`, `stride`, `kernel_size` and `dropout` apply only to the options that use them. The schema defaults are `normalization = "none"` and `temporal = "linear"`; the preset sets `normalization = "revin"`.
- **Loss.** Any point loss from `training.loss` is valid; the head has no loss of its own and the model defines no `training_objective` or auxiliary loss.
- **Smoke.** The spec declares no smoke config; `tsf model compose spec.toml --write-config run.toml --dataset <name> --smoke` writes a one-epoch CPU config.
- **Evidence.** Slot validation and adapter tests are in `tests/test_composed.py`.

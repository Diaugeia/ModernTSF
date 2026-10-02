---
name: "ARMD"
summary: "ARMD (Auto-Regressive Moving Diffusion) treats forecasting as a deterministic diffusion in which the future window is the initial state, the history window is the final state, and intermediate states are the length-T window slid between them. A linear devolution network with a learnable step-dependent weight predicts the future from any intermediate window; training matches the evolution trend implied by that prediction to the true one, and forecasting starts from the history window and applies deterministic DDIM updates in a few steps."
paper: "https://arxiv.org/abs/2412.09328"
paper_title: "Auto-Regressive Moving Diffusion Models for Time Series Forecasting"
venue: "AAAI 2025"
year: 2025
code: "https://github.com/daxin007/ARMD"
revision: "07a3dc4674f2f5cdf401e1a7f8895939bb2b07e2"
license: "MIT"
tagline: "Deterministic diffusion that slides a window from history to future; a linear devolution net, few-step DDIM forecast."
tags: ["diffusion", "linear", "channel-independent", "lightweight", "loss-framework", "time-series"]
composition: ["normalization=none", "decomposition=none", "temporal=local:linear-devolution-network", "channel=none", "head=local:ddim-window-sampler", "loss=local:evolution-trend-l1"]
---
# ARMD

## Key ideas

- `Model.intermediate_state` slides a length-T window (T = `pred_len`) over history then future, so the diffusion chain is deterministic and ends at the history window.
- `Model.devolve` is the linear devolution network: one `nn.Linear` over time shared by channels, mixed with the input by a learnable per-step weight `step_weight` (Eq. 5).
- The `training_objective` calls `Model.training_loss`: weighted L1 between the true and predicted evolution trends (Eq. 3, 6, 7) at one random diffusion step, replacing the configured loss.
- `Model.forward` starts from the last T history steps and applies deterministic DDIM updates with the predicted trend in place of noise, in `sampling_steps` steps (one step by default).

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2412.09328); title: Auto-Regressive Moving Diffusion Models for Time Series Forecasting; venue/year: AAAI 2025 / 2025
- [codebase](https://github.com/daxin007/ARMD); revision: `07a3dc4674f2f5cdf401e1a7f8895939bb2b07e2`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/ARMD.toml`](../../../../configs/models/ARMD.toml).

## Differences

Independent implementation; no official code was copied (official repository MIT, pinned revision above; checked `Models/autoregressive_diffusion/armd.py`, `Models/autoregressive_diffusion/linear.py`, `engine/solver.py`, `main.py`, the configs, and the paper's Eq. 1-10). Inputs are `[B, seq_len, enc_in]` with `seq_len >= pred_len`, marks and decoder inputs are ignored, and the output is the `[B, pred_len, enc_in]` point forecast (the forecast is deterministic, so no distribution output is declared). No shared component matches: the schedules, the devolution network and the sampler are model-local.

- **History length.** The paper and official code use history length equal to the horizon T (a 2T window); the model takes the last `pred_len` steps of a longer history (the official sampler slices the first `pred_len` steps of a 2T window, which is the history for `seq_len == pred_len`) and requires `seq_len >= pred_len`.
- **Diffusion length.** Official code hard-codes 96 steps in the network schedules and a global `pred_len = 96`; here the number of diffusion steps and every schedule length equal `pred_len`, as the paper states (maximum step equals the series length).
- **Input deviation.** The official network adds the training-time deviation in place on its input, which is a view of the training window, so the noise also leaks into the loss target and the intermediate state; here the deviation is applied to the network input only and the trend losses use the clean state and future, as in Eq. 3 and 6.
- **Loss.** The official loss adds a per-step weight `sqrt(alpha) sqrt(1 - abar) / beta / 100` that Eq. 7 omits; it is reproduced. One diffusion step is drawn per batch, as officially. `loss_type` `l1` or `l2` and `w_grad` follow the official configs (`w_grad=False` is used for ETTm1).
- **Sampling.** Only the deterministic DDIM update of Eq. 9-10 with `sampling_steps` steps is provided. Official code uses it when `sampling_timesteps < timesteps` and otherwise the stochastic ancestral sampler with clamping to `[-1, 1]`, which is not reproduced; the official configs use 1 or 2 steps.
- **Data and training.** Official data are min-max scaled to `[-1, 1]` over 2T windows with EMA weights (decay 0.995), gradient clipping and a plateau scheduler; here the repository's standard scaling, trainer and optimiser configuration apply and there is no EMA.
- **Evidence.** Structure and reference-formula tests are in `tests/test_armd.py`; unified verification evidence has not been produced yet and is left stale for CI.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `sampling_steps=1`, `loss_type='l1'`, `beta_schedule='cosine'`, `w_grad=True`
<!-- model-card:canonical:end -->

## Verification

Independent implementation; no official code was copied (official repository MIT, pinned revision above; checked `Models/autoregressive_diffusion/armd.py`, `Models/autoregressive_diffusion/linear.py`, `engine/solver.py`, `main.py`, the configs, and the paper's Eq. 1-10). Inputs are `[B, seq_len, enc_in]` with `seq_len >= pred_len`, marks and decoder inputs are ignored, and the output is the `[B, pred_len, enc_in]` point forecast (the forecast is deterministic, so no distribution output is declared). No shared component matches: the schedules, the devolution network and the sampler are model-local.

- **History length.** The paper and official code use history length equal to the horizon T (a 2T window); the model takes the last `pred_len` steps of a longer history (the official sampler slices the first `pred_len` steps of a 2T window, which is the history for `seq_len == pred_len`) and requires `seq_len >= pred_len`.
- **Diffusion length.** Official code hard-codes 96 steps in the network schedules and a global `pred_len = 96`; here the number of diffusion steps and every schedule length equal `pred_len`, as the paper states (maximum step equals the series length).
- **Input deviation.** The official network adds the training-time deviation in place on its input, which is a view of the training window, so the noise also leaks into the loss target and the intermediate state; here the deviation is applied to the network input only and the trend losses use the clean state and future, as in Eq. 3 and 6.
- **Loss.** The official loss adds a per-step weight `sqrt(alpha) sqrt(1 - abar) / beta / 100` that Eq. 7 omits; it is reproduced. One diffusion step is drawn per batch, as officially. `loss_type` `l1` or `l2` and `w_grad` follow the official configs (`w_grad=False` is used for ETTm1).
- **Sampling.** Only the deterministic DDIM update of Eq. 9-10 with `sampling_steps` steps is provided. Official code uses it when `sampling_timesteps < timesteps` and otherwise the stochastic ancestral sampler with clamping to `[-1, 1]`, which is not reproduced; the official configs use 1 or 2 steps.
- **Data and training.** Official data are min-max scaled to `[-1, 1]` over 2T windows with EMA weights (decay 0.995), gradient clipping and a plateau scheduler; here the repository's standard scaling, trainer and optimiser configuration apply and there is no EMA.
- **Evidence.** Structure and reference-formula tests are in `tests/test_armd.py`; unified verification evidence has not been produced yet and is left stale for CI.

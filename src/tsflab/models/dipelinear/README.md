---
name: "DiPELinear"
summary: "DiPE-Linear replaces a dense history-to-horizon linear map with a disentangled chain: a static zero-phase frequency attention, a static per-time-step attention, and an independent per-frequency complex mapping computed as one long FFT convolution. Low-rank weight sharing mixes a few weight sets across channels, and the SFALoss weights a frequency L1 term by the learned frequency attention."
paper: "https://arxiv.org/abs/2411.17257"
paper_title: "Disentangled Parameter-Efficient Linear Model for Long-Term Time Series Forecasting"
venue: "arXiv preprint"
year: 2024
code: "https://github.com/wintertee/DiPE-Linear"
revision: "b26e25015fba2c6a32e1f4eae8d337ccd9fe4067"
license: "MIT"
tagline: "Static frequency and time attention, then per-frequency complex FFT convolution to the horizon; low-rank sharing."
tags: ["linear", "frequency", "channel-independent", "lightweight", "normalization", "loss-framework", "time-series"]
composition: ["normalization=local:instance-mean-std", "decomposition=local:static-frequential-attention", "temporal=component:fft_extrapolation_conv", "channel=component:weight_set_router", "head=local:static-temporal-attention-and-denorm", "loss=local:sfa-loss"]
---
# DiPELinear

## Key ideas

- Static Frequential Attention (SFA): a learned zero-phase filter, `irfft(theta_sfa * rfft(x))`, followed by Static Temporal Attention (STA), a learned per-step scaling of the filtered history (`Model.forward`).
- Independent Frequential Mapping (IFM) is the `fft_extrapolation_conv` component: the history is zero padded to history plus horizon, every rfft bin gets its own complex weight and bias, and the last `pred_len` samples are the forecast (an `O(L log L)` kernel as long as the series).
- Low-rank weight sharing: `num_experts` weight sets per module are mixed per channel by `softmax(R / tau)` (`weight_set_router`), with the temperature annealed from 30 to 1 over the first 10 epochs.
- SFALoss (`Model.sfa_loss`) mixes a time-domain MSE with an L1 error between ortho-normalized rfft spectra weighted by the detached SFA map; it is wired as the runner's `training_objective` and replaces the configured criterion while training.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2411.17257); title: Disentangled Parameter-Efficient Linear Model for Long-Term Time Series Forecasting; venue/year: arXiv preprint / 2024
- [codebase](https://github.com/wintertee/DiPE-Linear); revision: `b26e25015fba2c6a32e1f4eae8d337ccd9fe4067`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/DiPELinear.toml`](../../../../configs/models/DiPELinear.toml).

## Differences

- Implementation: local rewrite from Sections 3.2 to 3.6 (Eqs. 1 to 11). The official code (`wintertee/DiPE-Linear`, MIT) was inspected at revision `b26e25015fba2c6a32e1f4eae8d337ccd9fe4067` (`timeprophet/models/DiPE.py`, `timeprophet/utils/callbacks.py`, `scripts/DiPE.sh`) to resolve omissions; no source was copied. `tests/fixtures/dipelinear_reference.pt` holds outputs and SFALoss values produced by the official module on seeded random weights (single expert, three experts, history shorter and longer than the horizon); `tests/test_dipelinear.py` reproduces them to 1e-5.
- Resolved from the official code, not stated in the paper: the guard band of about 1% of `L + H - 1` zeros on each side before the rfft; the initialization (SFA/STA weights 1, IFM weight 1 at the DC bin and 0 elsewhere, bias 0, routing matrix standard normal); dropout 0.1 after SFA; mean/std instance normalization with the unbiased standard deviation clamped at 1e-7 (kept local because `revin` uses the biased variance plus epsilon); the frequency-loss weights being normalized to unit mean and, when the history and horizon spectra have different lengths, truncated or zero padded to the horizon's bins; and the time-domain loss being the plain mean over all elements.
- Differences from the paper: SFALoss normalizes by the mean absolute weight, which equals the paper's L1-norm form for positive weights (the official code divides by the signed mean). Only the paper's routed formulation is provided; the official code's `individual_f/t/c` switches and its `t_loss` option (MAE time loss) are omitted, and the paper's SFALoss time term is always MSE.
- Differences from the official code: the temperature schedule is driven from `ModelSpec.training_setup` (records the training steps per epoch) and a step counter buffer advanced by the training objective, so it follows the official linear anneal per epoch without runner callbacks and resumes with checkpoints; in eval mode the temperature is the final value (the official code also uses it for testing but keeps the last training temperature during validation). The default preset uses `loss_alpha=0.9` (one of the values the official scripts use) and `num_experts=1`; per-dataset hyperparameters (for example `num_experts=4` for Electricity and Weather, `use_revin=False` there) and the 720-step lookback are not preset. With `MS` features the loss uses the trailing channels' weights. Reported benchmark numbers are not reproduction claims of this implementation.
- Verification: unified evidence plus the seeded official-output fixture above; no training was run.

## Shared components

- [`fft_extrapolation_conv`](../_components/fft_extrapolation_conv/README.md)
- [`weight_set_router`](../_components/weight_set_router/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `num_experts=1`, `use_revin=True`, `use_time_w=True`, `use_freq_w=True`, `loss_alpha=0.9`, `dropout=0.1`, `temperature_start=30.0`, `temperature_end=1.0`, `anneal_epochs=10`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Disentangled Parameter-Efficient Linear Model for Long-Term Time Series Forecasting
- **Venue**: arXiv preprint (v2, 2026)
- **Published**: 2024 (arXiv: 2024-11)
- **arXiv**: https://arxiv.org/abs/2411.17257

## Source and verification

- Implementation: local rewrite from Sections 3.2 to 3.6 (Eqs. 1 to 11). The official code (`wintertee/DiPE-Linear`, MIT) was inspected at revision `b26e25015fba2c6a32e1f4eae8d337ccd9fe4067` (`timeprophet/models/DiPE.py`, `timeprophet/utils/callbacks.py`, `scripts/DiPE.sh`) to resolve omissions; no source was copied. `tests/fixtures/dipelinear_reference.pt` holds outputs and SFALoss values produced by the official module on seeded random weights (single expert, three experts, history shorter and longer than the horizon); `tests/test_dipelinear.py` reproduces them to 1e-5.
- Resolved from the official code, not stated in the paper: the guard band of about 1% of `L + H - 1` zeros on each side before the rfft; the initialization (SFA/STA weights 1, IFM weight 1 at the DC bin and 0 elsewhere, bias 0, routing matrix standard normal); dropout 0.1 after SFA; mean/std instance normalization with the unbiased standard deviation clamped at 1e-7 (kept local because `revin` uses the biased variance plus epsilon); the frequency-loss weights being normalized to unit mean and, when the history and horizon spectra have different lengths, truncated or zero padded to the horizon's bins; and the time-domain loss being the plain mean over all elements.
- Differences from the paper: SFALoss normalizes by the mean absolute weight, which equals the paper's L1-norm form for positive weights (the official code divides by the signed mean). Only the paper's routed formulation is provided; the official code's `individual_f/t/c` switches and its `t_loss` option (MAE time loss) are omitted, and the paper's SFALoss time term is always MSE.
- Differences from the official code: the temperature schedule is driven from `ModelSpec.training_setup` (records the training steps per epoch) and a step counter buffer advanced by the training objective, so it follows the official linear anneal per epoch without runner callbacks and resumes with checkpoints; in eval mode the temperature is the final value (the official code also uses it for testing but keeps the last training temperature during validation). The default preset uses `loss_alpha=0.9` (one of the values the official scripts use) and `num_experts=1`; per-dataset hyperparameters (for example `num_experts=4` for Electricity and Weather, `use_revin=False` there) and the 720-step lookback are not preset. With `MS` features the loss uses the trailing channels' weights. Reported benchmark numbers are not reproduction claims of this implementation.
- Verification: unified evidence plus the seeded official-output fixture above; no training was run.

## Citation

```bibtex
@misc{zhao2024dipelinear,
  title         = {Disentangled Parameter-Efficient Linear Model for Long-Term Time Series Forecasting},
  author        = {Yuang Zhao and Tianyu Li and Jiadong Chen and Shenrong Ye and Fuxin Jiang and Xiaofeng Gao},
  year          = {2024},
  eprint        = {2411.17257},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2411.17257}
}
```

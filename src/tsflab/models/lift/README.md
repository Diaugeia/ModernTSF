---
name: "LIFT"
summary: "LIFT is a plug-in that refines a backbone's multivariate forecast with leading indicators. A non-parametric lead estimator finds, for every variate, the variates that lead it by some number of steps (largest lagged cross-correlation), the leaders are shifted into alignment with the horizon, and an adaptive frequency mixer with state-dependent filters merges them into the normalized forecast. This entry pairs the refiner with the DLinear backbone used in the official examples."
paper: "https://arxiv.org/abs/2401.17548"
paper_title: "Rethinking Channel Dependence for Multivariate Time Series Forecasting: Learning from Leading Indicators"
venue: "ICLR 2024"
year: 2024
code: "https://github.com/SJTU-DMTai/LIFT"
revision: "79cf8f157bb9a616c9733daf1e9c2cfefdc98c94"
license: "NOASSERTION"
tagline: "DLinear forecast refined by FFT-estimated, lag-shifted leading variates through state-gated frequency filters."
tags: ["linear", "plug-in", "channel-mixing", "lead-lag", "frequency", "lightweight"]
composition: ["normalization=local:instance-normalization-without-affine", "decomposition=none", "temporal=component:dlinear", "channel=local:lead-lag-shifted-leaders", "head=local:adaptive-frequency-mixer", "loss=loss:mse"]
---
# LIFT

## Key ideas

- `Model.estimate_leaders` computes circular cross-correlations of all variate pairs by FFT, keeps lag-local maxima, and picks the top-K leaders with their leading steps and signs (O(L log L) per pair).
- `Model.shift_leaders` shifts each leader by its leading step over the observed window concatenated with the preliminary forecast, and flips negatively correlated leaders.
- `Model.lead_filters` and `Model.refine` generate 2K+1 frequency filters from the leader correlations and a per-variate state distribution, filter the spectra of the forecast, leaders, and leader differences, and mix them with a complex linear map as a residual on the normalized forecast.
- The backbone is the `dlinear` component; the refiner is trained jointly with it.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2401.17548); title: Rethinking Channel Dependence for Multivariate Time Series Forecasting: Learning from Leading Indicators; venue/year: ICLR 2024 / 2024
- [codebase](https://github.com/SJTU-DMTai/LIFT); revision: `79cf8f157bb9a616c9733daf1e9c2cfefdc98c94`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/LIFT.toml`](../../../../configs/models/LIFT.toml).

## Differences

Independent implementation from the paper; no official code was copied. The official repository (`SJTU-DMTai/LIFT`, formerly `SJTU-Quant/LIFT`, no license file, hence `NOASSERTION`) was read at the pinned revision (`models/LIFT.py`, `util/lead_estimate.py`, `models/DLinear.py`) as reference only. Inputs are `[B, seq_len, enc_in]`; marks and decoder inputs are ignored; the output is `[B, pred_len, enc_in]`. `seq_len` must be at least 4, `enc_in` must equal the channel count, and `kernel_size` must be odd (it is the DLinear moving-average width).

- **Backbone.** The backbone is fixed to the shared DLinear component (shared weights across channels, `kernel_size` 25 by default) and trained jointly with the refiner. Official LIFT wraps any backbone and recommends a pretrained, frozen backbone (`--pretrain --freeze`); neither a frozen backbone nor other backbones are provided here.
- **Online lead estimation.** Official code precomputes leaders, leading steps and correlations over the whole dataset (`prefetch/`) and caches frozen-backbone predictions; here they are estimated online from each normalized lookback window with the same rules (circular FFT cross-correlation, local maxima only, lag 0 excluded, top-K by absolute correlation, step offset +1, sign flip for negative correlation, a variate may lead itself). Like the official code, the correlations are evaluated in chunks of `lead_chunk_size` target variates (default 32), so the full `[B, C, C, L]` correlation tensor is never materialized (peak `[B, chunk, C, L]`); the result is identical for any chunk size.
- **Filter weights.** The constant-one logit that competes with the `|corr|` logits in the temperature softmax follows the official code rather than the paper text. The official README notes the method was slightly revised after the paper submission.
- **State filters.** With `state_num=1` official code uses a state-free linear filter factory; here the state classifier branch always runs (a softmax over one state is constant, so the function is the same with unused parameters).
- **Initialisation.** The complex mixing map uses uniform real and imaginary parts bounded by `1/sqrt(in_features)`; the official one uses a complex `nn.Linear` initialisation (or a kaiming-initialised `ComplexLinear` under distributed training). The state prior, state bias and filter factory use the official bounds.
- **Training.** Loss, optimiser, schedule, early stopping and per-dataset hyperparameters are runner configuration, not model facts. The preset `leader_num=4` and `state_num=8` match the official README example.
- **Evidence.** Structure and reference-formula tests are in `tests/test_lift.py`.

## Shared components

- [`dlinear`](../_components/dlinear/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `leader_num=4`, `state_num=8`, `temperature=1.0`, `kernel_size=25`
<!-- model-card:canonical:end -->

## Source and verification

Independent implementation from the paper; no official code was copied. The official repository (`SJTU-DMTai/LIFT`, formerly `SJTU-Quant/LIFT`, no license file, hence `NOASSERTION`) was read at the pinned revision (`models/LIFT.py`, `util/lead_estimate.py`, `models/DLinear.py`) as reference only. Inputs are `[B, seq_len, enc_in]`; marks and decoder inputs are ignored; the output is `[B, pred_len, enc_in]`. `seq_len` must be at least 4, `enc_in` must equal the channel count, and `kernel_size` must be odd (it is the DLinear moving-average width).

- **Backbone.** The backbone is fixed to the shared DLinear component (shared weights across channels, `kernel_size` 25 by default) and trained jointly with the refiner. Official LIFT wraps any backbone and recommends a pretrained, frozen backbone (`--pretrain --freeze`); neither a frozen backbone nor other backbones are provided here.
- **Online lead estimation.** Official code precomputes leaders, leading steps and correlations over the whole dataset (`prefetch/`) and caches frozen-backbone predictions; here they are estimated online from each normalized lookback window with the same rules (circular FFT cross-correlation, local maxima only, lag 0 excluded, top-K by absolute correlation, step offset +1, sign flip for negative correlation, a variate may lead itself). Like the official code, the correlations are evaluated in chunks of `lead_chunk_size` target variates (default 32), so the full `[B, C, C, L]` correlation tensor is never materialized (peak `[B, chunk, C, L]`); the result is identical for any chunk size.
- **Filter weights.** The constant-one logit that competes with the `|corr|` logits in the temperature softmax follows the official code rather than the paper text. The official README notes the method was slightly revised after the paper submission.
- **State filters.** With `state_num=1` official code uses a state-free linear filter factory; here the state classifier branch always runs (a softmax over one state is constant, so the function is the same with unused parameters).
- **Initialisation.** The complex mixing map uses uniform real and imaginary parts bounded by `1/sqrt(in_features)`; the official one uses a complex `nn.Linear` initialisation (or a kaiming-initialised `ComplexLinear` under distributed training). The state prior, state bias and filter factory use the official bounds.
- **Training.** Loss, optimiser, schedule, early stopping and per-dataset hyperparameters are runner configuration, not model facts. The preset `leader_num=4` and `state_num=8` match the official README example.
- **Evidence.** Structure and reference-formula tests are in `tests/test_lift.py`.

## Citation

```bibtex
@inproceedings{LIFT,
  title     = {Rethinking Channel Dependence for Multivariate Time Series Forecasting: Learning from Leading Indicators},
  author    = {Lifan Zhao and Yanyan Shen},
  booktitle = {The Twelfth International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=JiTVtCUOpS}
}
```

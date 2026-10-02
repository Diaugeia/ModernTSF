---
name: "KalmanFilterTS"
summary: "KalmanFilterTS is a differentiable fixed-gain alpha-beta filter for a constant-velocity state, with bounded learnable gains per channel."
paper: "https://doi.org/10.1115/1.3662552"
paper_title: "A New Approach to Linear Filtering and Prediction Problems"
venue: "Journal of Basic Engineering"
year: 1960
tagline: "Differentiable alpha-beta level/velocity filter with learned per-channel gains and linear extrapolation."
tags: ["statistical", "state-space-filter", "channel-independent", "baseline", "lightweight"]
composition: ["normalization=none", "decomposition=none", "temporal=local:alpha-beta-constant-velocity-filter", "channel=local:channel-independent-learned-gains", "head=local:linear-extrapolation-of-level-and-velocity", "loss=loss:mse"]
---
# KalmanFilterTS

## Key ideas

- Runs a recursive level/velocity update over the window with innovation gains `alpha` and `beta` (sigmoid of learnable logits, one pair per channel).
- The forecast extrapolates `level + h * velocity` for horizon step `h`.
- It is a fixed-gain specialization with no covariance recursion, unit time steps, and only `2 * enc_in` parameters.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://doi.org/10.1115/1.3662552); title: A New Approach to Linear Filtering and Prediction Problems; venue/year: Journal of Basic Engineering / 1960
- codebase: not available

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/KalmanFilterTS.toml`](../../../../configs/models/KalmanFilterTS.toml).

## Differences

This clean-room baseline is a fixed-gain alpha-beta specialization, not the cited paper's full covariance-based Kalman filter. It learns gains directly and assumes unit time steps, a constant-velocity state, and no exogenous control. No third-party implementation was inspected or copied.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `initial_alpha=0.5`, `initial_beta=0.25`
<!-- model-card:canonical:end -->

## Paper
- **Title**: A New Approach to Linear Filtering and Prediction Problems
- **Venue**: Journal of Basic Engineering
- **Published**: 1960
- **Link**: https://doi.org/10.1115/1.3662552

## Abstract
Kalman filtering recursively predicts and corrects a latent state. The local model is the narrower fixed-gain alpha-beta variant: it tracks level and velocity with learned bounded gains and does not propagate a covariance matrix.

## In TSFLab
Default config: `configs/models/KalmanFilterTS.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

This clean-room baseline is a fixed-gain alpha-beta specialization, not the cited paper's full covariance-based Kalman filter. It learns gains directly and assumes unit time steps, a constant-velocity state, and no exogenous control. No third-party implementation was inspected or copied.

## Citation

```bibtex
@article{kalman1960new,
  author  = {Rudolf E. Kalman},
  title   = {A New Approach to Linear Filtering and Prediction Problems},
  journal = {Journal of Basic Engineering},
  volume  = {82},
  number  = {1},
  pages   = {35--45},
  year    = {1960},
  doi     = {10.1115/1.3662552}
}
```

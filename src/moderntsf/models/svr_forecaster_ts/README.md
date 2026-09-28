---
name: "SVRForecasterTS"
summary: "SVRForecasterTS is a differentiable RBF-basis epsilon-regression adaptation with learned support centres and an explicit epsilon-insensitive loss helper."
paper: "https://papers.nips.cc/paper/1996/hash/d38901788c533e8286cb6400b40b386d-Abstract.html"
paper_title: "Support Vector Regression Machines"
venue: "Advances in Neural Information Processing Systems 9"
year: 1996
---
# SVRForecasterTS

SVRForecasterTS is a differentiable RBF-basis epsilon-regression adaptation with learned support centres and an explicit epsilon-insensitive loss helper.

<!-- model-card:canonical:start -->
## Method overview

SVRForecasterTS is a differentiable RBF-basis epsilon-regression adaptation with learned support centres and an explicit epsilon-insensitive loss helper.

## Core architecture

SVRForecasterTS is a differentiable RBF-basis epsilon-regression adaptation with learned support centres and an explicit epsilon-insensitive loss helper.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://papers.nips.cc/paper/1996/hash/d38901788c533e8286cb6400b40b386d-Abstract.html); title: Support Vector Regression Machines; venue/year: Advances in Neural Information Processing Systems 9 / 1996
- codebase: not available

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/SVRForecasterTS.toml`](../../../../configs/models/SVRForecasterTS.toml).

## Differences

This is a clean-room differentiable adaptation, not a classical convex SVR solver: support centres and coefficients are optimized directly, there is no dual constrained optimization, and the standard repository trainer only uses epsilon loss when explicitly configured to call the helper. There is no residual linear head. No third-party implementation was inspected or copied.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `num_support=16`, `kernel_gamma=0.1`, `epsilon=0.1`, `l2_penalty=0.0001`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Support Vector Regression Machines
- **Venue**: Advances in Neural Information Processing Systems 9
- **Published**: 1996
- **Link**: https://papers.nips.cc/paper/1996/hash/d38901788c533e8286cb6400b40b386d-Abstract.html

## Abstract
Support Vector Regression uses an epsilon-insensitive objective and a kernel expansion. The local adaptation retains those two ideas but directly learns RBF centres and coefficients by gradient descent instead of solving the constrained dual problem.

## In ModernTSF
Default config: `configs/models/SVRForecasterTS.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

This is a clean-room differentiable adaptation, not a classical convex SVR solver: support centres and coefficients are optimized directly, there is no dual constrained optimization, and the standard repository trainer only uses epsilon loss when explicitly configured to call the helper. There is no residual linear head. No third-party implementation was inspected or copied.

## Citation

```bibtex
@inproceedings{drucker1996support,
  author    = {Harris Drucker and Christopher J. C. Burges and Linda Kaufman and Alexander J. Smola and Vladimir Vapnik},
  title     = {Support Vector Regression Machines},
  booktitle = {Advances in Neural Information Processing Systems 9 (NIPS 1996)},
  pages     = {155--161},
  year      = {1996},
  url       = {https://proceedings.neurips.cc/paper/1996/hash/d38901788c533e8286cb6400b40b386d-Abstract.html}
}
```

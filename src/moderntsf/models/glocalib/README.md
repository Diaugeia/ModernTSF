---
name: "GlocalIB"
summary: "Glocal-IB is a plug-in regularizer that aligns the latent embeddings of two views of a series through a global-local Information Bottleneck: a projector on one branch is pulled toward a stop-gradient embedding of the other branch, improving representation quality. It is originally a **time-series imputation** method (masked view vs complete view)."
paper: "https://arxiv.org/abs/2510.04910"
paper_title: "Glocal Information Bottleneck for Time Series Imputation"
venue: "NeurIPS 2025"
year: 2025
code: "https://github.com/Muyiiiii/NeurIPS-25-Glocal-IB"
revision: "1ee232e6d6b28329010db0305899511cb7fc9016"
license: "NOASSERTION"
---
# GlocalIB

Glocal-IB is a plug-in regularizer that aligns the latent embeddings of two
views of a series through a global-local Information Bottleneck: a projector on
one branch is pulled toward a stop-gradient embedding of the other branch,
improving representation quality. It is originally a **time-series imputation**
method (masked view vs complete view).

<!-- model-card:canonical:start -->
## Method overview

Glocal-IB is a plug-in regularizer that aligns the latent embeddings of two views of a series through a global-local Information Bottleneck: a projector on one branch is pulled toward a stop-gradient embedding of the other branch, improving representation quality.

## Core architecture

It is originally a **time-series imputation** method (masked view vs complete view).

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2510.04910); title: Glocal Information Bottleneck for Time Series Imputation; venue/year: NeurIPS 2025 / 2025
- [codebase](https://github.com/Muyiiiii/NeurIPS-25-Glocal-IB); revision: `1ee232e6d6b28329010db0305899511cb7fc9016`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/GlocalIB.toml`](../../../../configs/models/GlocalIB.toml).

## Differences

Pinned source inspection: `Glocal_IB/Glocal_IB.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed. Reference-only source code was not copied.

- Independent local implementation from paper equations (6)-(8),
  (12)-(14); the unlicensed repository is reference-only and was inspected at the pinned revision
  as implementation material.
- Forecasting adaptation only; no imputation benchmark/result reference comparison claim.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `align_weight=0.5`, `mask_ratio=0.25`, `align_loss_type='cos_align'`, `kl_weight=0.01`
<!-- model-card:canonical:end -->

## In ModernTSF
Default config: `configs/models/GlocalIB.toml`; specification: `spec.py`;
runtime module: `model.py`.

**Forecasting design** (ModernTSF is forecasting-only, no missingness). The
alignment mechanism is kept faithful and the two views are adapted: the **clean
lookback** `x` is the anchor (it always exists, so it produces the forecast and
its embedding is the detached alignment target), and an **augmented copy**
`x_aug` (random temporal masking, training-only) is the corrupted view whose
projected embedding is pulled toward the anchor. The implementation is
independently written from the paper equations (no PyPOTS/PyGrinder dependency).

Objective: `L = L_pred(Ŷ, Y) + kl_weight · KL(q(z|x)||N(0,I)) + align_weight · L_align`.
The alignment term needs only `x`, so it rides the trainer's `aux_loss`
convention; eval is a plain single forward. Key params: `d_model`,
`align_weight`, `mask_ratio`, `align_loss_type`. Verify with
`uv run tsf smoke --model GlocalIB`.

Official reference reference: https://github.com/Muyiiiii/NeurIPS-25-Glocal-IB

## Source and verification

Pinned source inspection: `Glocal_IB/Glocal_IB.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

Local implementation: confirmed. Reference-only source code was not copied.

- Independent local implementation from paper equations (6)-(8),
  (12)-(14); the unlicensed repository is reference-only and was inspected at the pinned revision
  as implementation material.
- Forecasting adaptation only; no imputation benchmark/result reference comparison claim.

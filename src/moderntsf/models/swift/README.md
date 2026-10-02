---
name: "SWIFT"
summary: "SWIFT reversibly normalizes each instance (RevIN), maps every channel's sub-series with a single-level, lossless Haar wavelet decomposition into an approximation and a detail band, fuses those two bands with a small shared depthwise convolution (cross-band information fusion), maps the fused coefficients from input-length to horizon-length with a single linear layer or a shallow MLP (the sub-series mapper), and reconstructs the forecast with the matching inverse Haar transform before denormalizing."
paper: "https://arxiv.org/abs/2501.16178"
paper_title: "SWIFT: Mapping Sub-series with Wavelet Decomposition Improves Time Series Forecasting"
venue: "arXiv"
year: 2025
code: "https://github.com/Lancelot-Xie/SWIFT"
revision: "f1f4be7c4eeae09e75749b22090bc1b21cf33641"
license: "unlicensed (repository README displays an MIT badge but no LICENSE file is present in the tree; inspected only for read-only paper-structure clarification, no source copied)"

---
# SWIFT

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2501.16178); title: SWIFT: Mapping Sub-series with Wavelet Decomposition Improves Time Series Forecasting; venue/year: arXiv / 2025
- [codebase](https://github.com/Lancelot-Xie/SWIFT); revision: `f1f4be7c4eeae09e75749b22090bc1b21cf33641`; license: `unlicensed (repository README displays an MIT badge but no LICENSE file is present in the tree; inspected only for read-only paper-structure clarification, no source copied)`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/SWIFT.toml`](../../../../configs/models/SWIFT.toml).

## Differences

Inspected official files: `models/SWIFT_Linear.py`, `models/SWIFT_MLP.py`,
`layers/RevIN.py`, at revision `f1f4be7c4eeae09e75749b22090bc1b21cf33641`.

**Paper-driven local implementation.** The official code depends on the
external `pytorch_wavelets` package (`DWT1DForward`/`DWT1DInverse`, Haar
wavelet, `J=1`) for the wavelet transform; the local `haar_dwt1d` component
reimplements the exact orthonormal single-level Haar decomposition/reconstruction
in closed form (no external wavelet dependency), which is lossless and
algebraically identical for even-length axes and additionally supports
odd-length axes via a documented replicate-pad-and-truncate convention that
the official code does not need to specify at the paper's benchmark lengths.
The official repository also exposes a `no_revin` flag that swaps RevIN for a
plain mean-subtraction/no rescale-back path; this local implementation always
uses `RevIN`, matching the paper's reported default configuration, and drops
that flag for a smaller, single-path parameter surface. The external
repository is reference-only; no source file was copied or adapted.

## Shared components

- [`haar_dwt1d`](../_components/haar_dwt1d/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `hidden_size=0`, `conv_kernel=3`, `use_convdropout=False`, `conv_dropout=0.1`, `not_independent=False`
<!-- model-card:canonical:end -->

## Source and verification

Inspected official files: `models/SWIFT_Linear.py`, `models/SWIFT_MLP.py`,
`layers/RevIN.py`, at revision `f1f4be7c4eeae09e75749b22090bc1b21cf33641`.

**Paper-driven local implementation.** The official code depends on the
external `pytorch_wavelets` package (`DWT1DForward`/`DWT1DInverse`, Haar
wavelet, `J=1`) for the wavelet transform; the local `haar_dwt1d` component
reimplements the exact orthonormal single-level Haar decomposition/reconstruction
in closed form (no external wavelet dependency), which is lossless and
algebraically identical for even-length axes and additionally supports
odd-length axes via a documented replicate-pad-and-truncate convention that
the official code does not need to specify at the paper's benchmark lengths.
The official repository also exposes a `no_revin` flag that swaps RevIN for a
plain mean-subtraction/no rescale-back path; this local implementation always
uses `RevIN`, matching the paper's reported default configuration, and drops
that flag for a smaller, single-path parameter surface. The external
repository is reference-only; no source file was copied or adapted.

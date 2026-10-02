---
name: "AWEMixer"
summary: "AWEMixer adaptively weights undecimated wavelet subbands with a Frequency Router driven by four spectral/temporal descriptors, then lets multi-scale temporal anchors selectively absorb that weighted frequency context through a Coherent Gated Fusion cross-attention block before a linear forecast head."
paper: "https://arxiv.org/abs/2511.04722"
paper_title: "AWEMixer: Adaptive Wavelet-Enhanced Mixer Network for Long-Term Time Series Forecasting"
venue: "arXiv"
year: 2025
code: "https://github.com/hit636/AWEMixer"
revision: "8e660f93c2535e6eb64ba47647babf21f3dbacfa"
license: "Apache-2.0"
---
# AWEMixer

AWEMixer adaptively weights undecimated wavelet subbands with a Frequency Router driven by four spectral/temporal descriptors, then lets multi-scale temporal anchors selectively absorb that weighted frequency context through a Coherent Gated Fusion cross-attention block before a linear forecast head.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2511.04722); title: AWEMixer: Adaptive Wavelet-Enhanced Mixer Network for Long-Term Time Series Forecasting; venue/year: arXiv / 2025
- [codebase](https://github.com/hit636/AWEMixer); revision: `8e660f93c2535e6eb64ba47647babf21f3dbacfa`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/AWEMixer.toml`](../../../../configs/models/AWEMixer.toml).

## Differences

Clean-room implementation: confirmed. The Frequency Router descriptor
equations, the Coherent Gated Fusion Block, and the overall data flow were
re-derived from the pinned official file's module boundaries and comments
describing which paper equations they implement; no source lines were
copied.

- The official code builds its undecimated wavelet filters from `pywt`
  (dividing PyWavelets' already-orthonormal filters by an extra `sqrt(2)`,
  and reversing them). This implementation uses the cataloged
  `UndecimatedWaveletTransform`, whose filter taps are the standard
  orthonormal Daubechies coefficients without that extra rescaling; this is
  a fixed, learnable-linear-layer-absorbed scale difference, not a
  structural one, since every subband is immediately passed through a
  learned `Linear` embedding.
- The official FFT band-splitting loop computes band edges with a Python
  `for` loop of per-band start/end indices; this implementation computes the
  same evenly spaced edges with one `torch.linspace` call, which is
  numerically equivalent.
- `return_aux` (an optional auxiliary-output diagnostic mode in the official
  code) is not exposed; the equivalent tensors (`last_router_weights`,
  `last_gates`) are always recorded as instance attributes instead, for
  parity with this catalog's structural-test conventions.

## Shared components

- [`revin`](../_components/revin/README.md)
- [`wavelet`](../_components/wavelet/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `d_model=128`, `dropout=0.2`, `num_scales=3`, `wavelet_level=3`, `wavelet='db4'`, `num_fusion_layers=1`
<!-- model-card:canonical:end -->

## Source and verification

Clean-room implementation: confirmed. The Frequency Router descriptor
equations, the Coherent Gated Fusion Block, and the overall data flow were
re-derived from the pinned official file's module boundaries and comments
describing which paper equations they implement; no source lines were
copied.

- The official code builds its undecimated wavelet filters from `pywt`
  (dividing PyWavelets' already-orthonormal filters by an extra `sqrt(2)`,
  and reversing them). This implementation uses the cataloged
  `UndecimatedWaveletTransform`, whose filter taps are the standard
  orthonormal Daubechies coefficients without that extra rescaling; this is
  a fixed, learnable-linear-layer-absorbed scale difference, not a
  structural one, since every subband is immediately passed through a
  learned `Linear` embedding.
- The official FFT band-splitting loop computes band edges with a Python
  `for` loop of per-band start/end indices; this implementation computes the
  same evenly spaced edges with one `torch.linspace` call, which is
  numerically equivalent.
- `return_aux` (an optional auxiliary-output diagnostic mode in the official
  code) is not exposed; the equivalent tensors (`last_router_weights`,
  `last_gates`) are always recorded as instance attributes instead, for
  parity with this catalog's structural-test conventions.

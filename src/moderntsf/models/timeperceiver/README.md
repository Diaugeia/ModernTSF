---
name: "TimePerceiver"
summary: "TimePerceiver is a time series forecasting model built around a Perceiver-style encoder-decoder architecture. It generalises the forecasting task to arbitrary temporal prediction objectives (extrapolation, interpolation, and imputation) by dividing sequences into patch tokens, encoding them through a set of latent bottleneck representations that interact with all input patches via cross-attention to capture both temporal and cross-channel dependencies, and decoding future patches with learnable queries corresponding to target timestamps. The design is paired with a unified training strategy that tightly aligns the encoder, decoder, and prediction objectives."
paper: "https://arxiv.org/abs/2512.22550"
paper_title: "TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting"
venue: "NeurIPS 2025"
year: 2025
code: "https://github.com/efficient-learning-lab/TimePerceiver"
revision: "7e30cc07b51c709f408409fd60a34c81ae8990be"
license: "MIT"
---
# TimePerceiver

TimePerceiver is a time series forecasting model built around a Perceiver-style encoder-decoder architecture. It generalises the forecasting task to arbitrary temporal prediction objectives (extrapolation, interpolation, and imputation) by dividing sequences into patch tokens, encoding them through a set of latent bottleneck representations that interact with all input patches via cross-attention to capture both temporal and cross-channel dependencies, and decoding future patches with learnable queries corresponding to target timestamps. The design is paired with a unified training strategy that tightly aligns the encoder, decoder, and prediction objectives.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2512.22550); title: TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting; venue/year: NeurIPS 2025 / 2025
- [codebase](https://github.com/efficient-learning-lab/TimePerceiver); revision: `7e30cc07b51c709f408409fd60a34c81ae8990be`; license: `MIT`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/TimePerceiver.toml`](../../../../configs/models/TimePerceiver.toml).

## Differences

Clean-room implementation: confirmed. The implementation was derived independently from the paper's input-segment encoder, latent bottleneck, latent self-processing, and timestamp-query decoder; reference source code was not copied or reused. The repository API provides past-to-future forecasting, not the paper's generalized interpolation/imputation training sampler.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=32`, `n_heads=2`, `patch_len=16`, `dropout=0.1`, `num_latents=8`, `latent_dim=128`, `latent_d_ff=256`, `num_latent_blocks=1`, `query_share=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2024-12)
- **arXiv**: https://arxiv.org/abs/2512.22550

## Abstract
In machine learning, effective modeling requires a holistic consideration of how to encode inputs, make predictions (i.e., decoding), and train the model. However, in time-series forecasting, prior work has predominantly focused on encoder design, often treating prediction and training as separate or secondary concerns. In this paper, we propose TimePerceiver, a unified encoder-decoder forecasting framework that is tightly aligned with an effective training strategy. To be specific, we first generalize the forecasting task to include diverse temporal prediction objectives such as extrapolation, interpolation, and imputation. Since this generalization requires handling input and target segments that are arbitrarily positioned along the temporal axis, we design a novel encoder-decoder architecture that can flexibly perceive and adapt to these varying positions. For encoding, we introduce a set of latent bottleneck representations that can interact with all input segments to jointly capture temporal and cross-channel dependencies. For decoding, we leverage learnable queries corresponding to target timestamps to effectively retrieve relevant information. Extensive experiments demonstrate that our framework consistently and significantly outperforms prior state-of-the-art baselines across a wide range of benchmark datasets. The code is available at https://github.com/efficient-learning-lab/TimePerceiver.

## In ModernTSF
Default config: `configs/models/TimePerceiver.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed. The implementation was derived independently from the paper's input-segment encoder, latent bottleneck, latent self-processing, and timestamp-query decoder; reference source code was not copied or reused. The repository API provides past-to-future forecasting, not the paper's generalized interpolation/imputation training sampler.

## Citation

```bibtex
@misc{lee2025timeperceiver,
  author        = {Jaebin Lee and
                  Hankook Lee},
  title         = {TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting},
  year          = {2025},
  eprint        = {2512.22550},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2512.22550}
}
```

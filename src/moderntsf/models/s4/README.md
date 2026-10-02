---
name: "S4"
summary: "S4 (Structured State Space Sequence model) is a general sequence model for time series forecasting that is built on the diagonal S4D variant of the structured state space framework. It uses an FFT-based long convolution kernel derived from a diagonalized state matrix, enabling efficient modeling of long-range dependencies without custom CUDA operators. In ModernTSF the S4D layers are stacked with residual connections over the time axis, preceded by an input projection and followed by a linear forecast head mapping the sequence length to the prediction horizon."
paper: "https://arxiv.org/abs/2111.00396"
paper_title: "Efficiently Modeling Long Sequences with Structured State Spaces"
venue: "ICLR 2022"
year: 2022
code: "https://github.com/state-spaces/s4"
revision: "e757cef57d89e448c413de7325ed5601aceaac13"
license: "Apache-2.0"
---
# S4

S4 (Structured State Space Sequence model) is a general sequence model for time series forecasting that is built on the diagonal S4D variant of the structured state space framework. It uses an FFT-based long convolution kernel derived from a diagonalized state matrix, enabling efficient modeling of long-range dependencies without custom CUDA operators. In ModernTSF the S4D layers are stacked with residual connections over the time axis, preceded by an input projection and followed by a linear forecast head mapping the sequence length to the prediction horizon.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2111.00396); title: Efficiently Modeling Long Sequences with Structured State Spaces; venue/year: ICLR 2022 / 2022
- [codebase](https://github.com/state-spaces/s4); revision: `e757cef57d89e448c413de7325ed5601aceaac13`; license: `Apache-2.0`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/S4.toml`](../../../../configs/models/S4.toml).

## Differences

**Clean-room implementation: confirmed.** Exact diagonal zero-order-hold discretization, impulse response, FFT convolution, gradients, and runtime contracts are checked locally. Inputs are `[B, seq_len, enc_in]`, outputs are `[B, pred_len, c_out]`, and marks are ignored. This is a diagonal S4D-style approximation rather than full NPLR S4; the linked codebase is reference-only and no source was copied.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `d_state=64`, `e_layers=2`, `dropout=0.1`, `use_norm=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Efficiently Modeling Long Sequences with Structured State Spaces
- **Venue**: ICLR 2022
- **Published**: 2022 (arXiv: 2021-11)
- **arXiv**: https://arxiv.org/abs/2111.00396

## Abstract
A central goal of sequence modeling is designing a single principled model that can address sequence data across a range of modalities and tasks, particularly on long-range dependencies. Although conventional models including RNNs, CNNs, and Transformers have specialized variants for capturing long dependencies, they still struggle to scale to very long sequences of 10000 or more steps. A promising recent approach proposed modeling sequences by simulating the fundamental state space model (SSM) x'(t) = Ax(t) + Bu(t), y(t) = Cx(t) + Du(t), and showed that for appropriate choices of the state matrix A, this system could handle long-range dependencies mathematically and empirically. However, this method has prohibitive computation and memory requirements, rendering it infeasible as a general sequence modeling solution. We propose the Structured State Space sequence model (S4) based on a new parameterization for the SSM, and show that it can be computed much more efficiently than prior approaches while preserving their theoretical strengths. Our technique involves conditioning A with a low-rank correction, allowing it to be diagonalized stably and reducing the SSM to the well-studied computation of a Cauchy kernel. S4 achieves strong empirical results across a diverse range of established benchmarks, including (i) 91% accuracy on sequential CIFAR-10 with no data augmentation or auxiliary losses, on par with a larger 2-D ResNet, (ii) substantially closing the gap to Transformers on image and language modeling tasks, while performing generation 60× faster (iii) SoTA on every task from the Long Range Arena benchmark, including solving the challenging Path-X task of length 16k that all prior work fails on, while being as efficient as all competitors.

## In ModernTSF
Default config: `configs/models/S4.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

**Clean-room implementation: confirmed.** Exact diagonal zero-order-hold discretization, impulse response, FFT convolution, gradients, and runtime contracts are checked locally. Inputs are `[B, seq_len, enc_in]`, outputs are `[B, pred_len, c_out]`, and marks are ignored. This is a diagonal S4D-style approximation rather than full NPLR S4; the linked codebase is reference-only and no source was copied.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/GuGR22,
  author       = {Albert Gu and
                  Karan Goel and
                  Christopher R{\'{e}}},
  title        = {Efficiently Modeling Long Sequences with Structured State Spaces},
  booktitle    = {The Tenth International Conference on Learning Representations, {ICLR}
                  2022, Virtual Event, April 25-29, 2022},
  publisher    = {OpenReview.net},
  year         = {2022},
  url          = {https://arxiv.org/abs/2111.00396},
  eprinttype   = {arXiv},
  eprint       = {2111.00396},
  timestamp    = {Sat, 20 Aug 2022 01:15:42 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/GuGR22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

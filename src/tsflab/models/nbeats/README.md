---
name: "NBeats"
summary: "N-BEATS is a deep neural architecture for univariate and multivariate time series point forecasting. It is built on a deep stack of fully-connected blocks, each producing a backcast (reconstruction of the input) and a forecast, linked by backward and forward residual connections. Two configurations exist: a generic version relying solely on deep learning primitives, and an interpretable version whose basis expansion functions correspond to trend and seasonality components."
paper: "https://arxiv.org/abs/1905.10437"
paper_title: "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"
venue: "ICLR 2020"
year: 2020
code: "https://github.com/philipperemy/n-beats"
revision: "06a4e209ada80bf1f403ced5228261784dfb26ed"
license: "MIT"
tagline: "Doubly residual stacks of FC blocks emitting backcast and forecast through trend, seasonality, or generic bases."
tags: ["mlp", "decomposition", "basis-expansion", "channel-independent", "interpretable"]
composition: ["normalization=none", "decomposition=local:trend-seasonality-basis-stacks", "temporal=local:fully-connected-residual-blocks", "channel=local:channel-independent-shared-weights", "head=local:basis-expansion-forecast", "loss=loss:mse"]
---
# NBeats

## Key ideas

- `NBeatsBlock` is a four-layer ReLU MLP whose two theta heads are projected by a basis: `trend_basis`, `seasonality_basis`, or a learned generic matrix.
- Blocks are chained with backward residual links (the input minus the backcast) and forward links (forecasts are summed).
- Default stacks are trend, seasonality, then generic (`stack_types`, `thetas_dim`); `share_weights_in_stack` shares blocks.
- Each channel is treated as a univariate series folded into the batch.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/1905.10437); title: N-BEATS: Neural basis expansion analysis for interpretable time series forecasting; venue/year: ICLR 2020 / 2020
- [codebase](https://github.com/philipperemy/n-beats); revision: `06a4e209ada80bf1f403ced5228261784dfb26ed`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/NBeats.toml`](../../../../configs/models/NBeats.toml).

## Differences

This implementation was rewritten for TSFLab after checking both the paper
and the pinned official open-source implementation. Each shared univariate block produces
backcast and forecast expansion coefficients; backcasts are subtracted from the
running residual and partial forecasts are summed. Generic blocks learn their
bases, trend blocks use polynomial bases, and seasonality blocks use Fourier
bases. The external repository is reference-only; no source file was copied or
adapted. Published benchmark reproduction is separate.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `stack_types=['trend', 'seasonality', 'generic']`, `nb_blocks_per_stack=3`, `thetas_dim=[4, 8, 8]`, `hidden_layer_units=256`, `share_weights_in_stack=False`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
- **Venue**: ICLR 2020
- **Published**: 2020 (arXiv: 2019-05)
- **arXiv**: https://arxiv.org/abs/1905.10437

## Abstract
We focus on solving the univariate times series point forecasting problem using deep learning. We propose a deep neural architecture based on backward and forward residual links and a very deep stack of fully-connected layers. The architecture has a number of desirable properties, being interpretable, applicable without modification to a wide array of target domains, and fast to train. We test the proposed architecture on several well-known datasets, including M3, M4 and TOURISM competition datasets containing time series from diverse domains. We demonstrate state-of-the-art performance for two configurations of N-BEATS for all the datasets, improving forecast accuracy by 11% over a statistical benchmark and by 3% over last year's winner of the M4 competition, a domain-adjusted hand-crafted hybrid between neural network and statistical time series models. The first configuration of our model does not employ any time-series-specific components and its performance on heterogeneous datasets strongly suggests that, contrarily to received wisdom, deep learning primitives such as residual blocks are by themselves sufficient to solve a wide range of forecasting problems. Finally, we demonstrate how the proposed architecture can be augmented to provide outputs that are interpretable without considerable loss in accuracy.

## In TSFLab
Default config: `configs/models/NBeats.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

This implementation was rewritten for TSFLab after checking both the paper
and the pinned official open-source implementation. Each shared univariate block produces
backcast and forecast expansion coefficients; backcasts are subtracted from the
running residual and partial forecasts are summed. Generic blocks learn their
bases, trend blocks use polynomial bases, and seasonality blocks use Fourier
bases. The external repository is reference-only; no source file was copied or
adapted. Published benchmark reproduction is separate.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/OreshkinCCB20,
  author       = {Boris N. Oreshkin and
                  Dmitri Carpov and
                  Nicolas Chapados and
                  Yoshua Bengio},
  title        = {{N-BEATS:} Neural basis expansion analysis for interpretable time
                  series forecasting},
  booktitle    = {8th International Conference on Learning Representations, {ICLR} 2020,
                  Addis Ababa, Ethiopia, April 26-30, 2020},
  publisher    = {OpenReview.net},
  year         = {2020},
  url          = {https://arxiv.org/abs/1905.10437},
  eprinttype   = {arXiv},
  eprint       = {1905.10437},
  timestamp    = {Thu, 07 May 2020 17:11:47 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/OreshkinCCB20.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

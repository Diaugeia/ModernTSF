---
name: "PAttn"
summary: "PAttn is a deliberately simple patch-based Transformer baseline for time-series forecasting, introduced in the NeurIPS 2024 Spotlight paper \"Are Language Models Actually Useful for Time Series Forecasting?\". It pads and unfolds the input into overlapping patches, linearly embeds each patch per channel, processes the patch tokens with a single self-attention encoder block, then flattens and linearly projects to the forecast horizon, demonstrating that this minimal architecture matches or exceeds much heavier LLM-based forecasters."
paper: "https://arxiv.org/abs/2406.16964"
paper_title: "Are Language Models Actually Useful for Time Series Forecasting?"
venue: "NeurIPS 2024"
year: 2024
code: "https://github.com/thuml/Time-Series-Library"
revision: "4e938a1767106324dd753b2a44832bf870a0252e"
license: "MIT"
---
# PAttn

PAttn is a deliberately simple patch-based Transformer baseline for time-series forecasting, introduced in the NeurIPS 2024 Spotlight paper "Are Language Models Actually Useful for Time Series Forecasting?". It pads and unfolds the input into overlapping patches, linearly embeds each patch per channel, processes the patch tokens with a single self-attention encoder block, then flattens and linearly projects to the forecast horizon, demonstrating that this minimal architecture matches or exceeds much heavier LLM-based forecasters.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2406.16964); title: Are Language Models Actually Useful for Time Series Forecasting?; venue/year: NeurIPS 2024 / 2024
- [codebase](https://github.com/thuml/Time-Series-Library); revision: `4e938a1767106324dd753b2a44832bf870a0252e`; license: `MIT`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/PAttn.toml`](../../../../configs/models/PAttn.toml).

## Differences

Clean-room implementation: confirmed. Reference-only source code was not copied.

Independent clean-room implementation from Figure 4 and appendix D.3. The
pinned repositories are reference-only and no source was copied.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `d_model=128`, `n_heads=8`, `patch_len=16`, `stride=8`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Are Language Models Actually Useful for Time Series Forecasting?
- **Venue**: NeurIPS 2024 (Spotlight)
- **Published**: 2024 (arXiv: 2024-06)
- **arXiv**: https://arxiv.org/abs/2406.16964

## Abstract
Large language models (LLMs) are being applied to time series forecasting. But are language models actually useful for time series? In a series of ablation studies on three recent and popular LLM-based time series forecasting methods, we find that removing the LLM component or replacing it with a basic attention layer does not degrade forecasting performance -- in most cases, the results even improve! We also find that despite their significant computational cost, pretrained LLMs do no better than models trained from scratch, do not represent the sequential dependencies in time series, and do not assist in few-shot settings. Additionally, we explore time series encoders and find that patching and attention structures perform similarly to LLM-based forecasters.

## In ModernTSF
Default config: `configs/models/PAttn.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed. Reference-only source code was not copied.

Independent clean-room implementation from Figure 4 and appendix D.3. The
pinned repositories are reference-only and no source was copied.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/TanMGAH24,
  author       = {Mingtian Tan and
                  Mike A. Merrill and
                  Vinayak Gupta and
                  Tim Althoff and
                  Tom Hartvigsen},
  editor       = {Amir Globersons and
                  Lester Mackey and
                  Danielle Belgrave and
                  Angela Fan and
                  Ulrich Paquet and
                  Jakub M. Tomczak and
                  Cheng Zhang},
  title        = {Are Language Models Actually Useful for Time Series Forecasting?},
  booktitle    = {Advances in Neural Information Processing Systems 37: Annual Conference
                  on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver,
                  BC, Canada, December 10 - 15, 2024},
  year         = {2024},
  url          = {http://papers.nips.cc/paper\_files/paper/2024/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract-Conference.html},
  timestamp    = {Tue, 26 May 2026 17:12:08 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/TanMGAH24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

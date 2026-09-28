---
name: "Pyraformer"
summary: "Pyraformer is a Transformer-based time series forecasting model that builds a multi-resolution pyramidal attention module (PAM) over the input sequence. Inter-scale tree connections summarize temporal features at progressively coarser resolutions, while intra-scale connections between neighboring tokens model dependencies at each resolution. This design achieves O(1) maximum signal-path length with respect to sequence length and linear time and space complexity, making it efficient for long-range forecasting on both single-step and multi-step horizons."
paper: "https://openreview.net/forum?id=0EXmFzUn5I"
paper_title: "Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting"
venue: "ICLR 2022"
year: 2022
code: "https://github.com/thuml/Time-Series-Library"
revision: "3a4819420d14095354aae96750ce8c499ef5f05e"
license: "MIT"
---
# Pyraformer

Pyraformer is a Transformer-based time series forecasting model that builds a multi-resolution pyramidal attention module (PAM) over the input sequence. Inter-scale tree connections summarize temporal features at progressively coarser resolutions, while intra-scale connections between neighboring tokens model dependencies at each resolution. This design achieves O(1) maximum signal-path length with respect to sequence length and linear time and space complexity, making it efficient for long-range forecasting on both single-step and multi-step horizons.

<!-- model-card:canonical:start -->
## Method overview

Pyraformer is a Transformer-based time series forecasting model that builds a multi-resolution pyramidal attention module (PAM) over the input sequence.

## Core architecture

Inter-scale tree connections summarize temporal features at progressively coarser resolutions, while intra-scale connections between neighboring tokens model dependencies at each resolution. This design achieves O(1) maximum signal-path length with respect to sequence length and linear time and space complexity, making it efficient for long-range forecasting on both single-step and multi-step horizons.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://openreview.net/forum?id=0EXmFzUn5I); title: Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting; venue/year: ICLR 2022 / 2022
- [codebase](https://github.com/thuml/Time-Series-Library); revision: `3a4819420d14095354aae96750ce8c499ef5f05e`; license: `MIT`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/Pyraformer.toml`](../../../../configs/models/Pyraformer.toml).

## Differences

Implementation: **clean-room rewrite** from the ICLR 2022 paper, especially
Equations (2) and (3) and the architecture in Figure 2. The local code was
written independently; source from the linked Time-Series-Library revision was
not copied and is retained only as a historical `reference-only` link.
Clean-room implementation: confirmed.

The rewrite constructs learned coarser temporal scales, explicitly builds the
paper's scale-local/child/parent PAM neighbourhoods, applies sparse multi-head
attention only across those edges, gathers the last observation's ancestor
chain, and projects it to the full forecast horizon. Raw ModernTSF marks are
accepted as six columns `[year, month, day, weekday, hour, minute]` and are
normalized locally before a learned calendar projection.

This is not a reproduction of the official training system. It uses direct
multi-horizon prediction strategy 1, learned strided convolutions for CSCM,
pre-normalized residual blocks, and dense PyTorch gather operations over a
padded sparse-neighbour table. It therefore preserves the defining pyramidal
graph but does not claim the paper's optimized kernel wall-clock complexity or
published numerical results.

The previous official reference candidate was blocked because the pinned implementation
passed `dropout` positionally into an embedding argument and expected
preprocessed time features rather than this repository's raw six-column marks.
That reference comparison blocker is resolved by replacement, not by pretending the two paths
are equivalent.

Configuration requires `d_model` to be divisible by `n_heads`, `inner_size` to
be odd, and `seq_len` to remain exactly divisible by every successive
`window_size` branching factor so the C-ary pyramid is unambiguous.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `n_heads=8`, `e_layers=2`, `d_ff=256`, `dropout=0.1`, `window_size=[4, 4]`, `inner_size=5`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting
- **Venue**: ICLR 2022 (Oral)
- **Published**: 2022
- **arXiv**: N/A

## Abstract
Accurate prediction of the future given the past based on time series data is of paramount importance, since it opens the door for decision making and risk management ahead of time. In practice, the challenge is to build a flexible but parsimonious model that can capture a wide range of temporal dependencies. In this paper, we propose Pyraformer by exploring the multiresolution representation of the time series. Specifically, we introduce the pyramidal attention module (PAM) in which the inter-scale tree structure summarizes features at different resolutions and the intra-scale neighboring connections model the temporal dependencies of different ranges. Under mild conditions, the maximum length of the signal traversing path in Pyraformer is a constant (i.e., O(1)) with regard to the sequence length L, while its time and space complexity scale linearly with L. Extensive numerical results show that Pyraformer typically achieves the highest prediction accuracy in both single-step and long-range forecasting tasks with the least amount of time and memory consumption, especially when the sequence is long.

## In ModernTSF
Default config: `configs/models/Pyraformer.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

Implementation: **clean-room rewrite** from the ICLR 2022 paper, especially
Equations (2) and (3) and the architecture in Figure 2. The local code was
written independently; source from the linked Time-Series-Library revision was
not copied and is retained only as a historical `reference-only` link.
Clean-room implementation: confirmed.

The rewrite constructs learned coarser temporal scales, explicitly builds the
paper's scale-local/child/parent PAM neighbourhoods, applies sparse multi-head
attention only across those edges, gathers the last observation's ancestor
chain, and projects it to the full forecast horizon. Raw ModernTSF marks are
accepted as six columns `[year, month, day, weekday, hour, minute]` and are
normalized locally before a learned calendar projection.

This is not a reproduction of the official training system. It uses direct
multi-horizon prediction strategy 1, learned strided convolutions for CSCM,
pre-normalized residual blocks, and dense PyTorch gather operations over a
padded sparse-neighbour table. It therefore preserves the defining pyramidal
graph but does not claim the paper's optimized kernel wall-clock complexity or
published numerical results.

The previous official reference candidate was blocked because the pinned implementation
passed `dropout` positionally into an embedding argument and expected
preprocessed time features rather than this repository's raw six-column marks.
That reference comparison blocker is resolved by replacement, not by pretending the two paths
are equivalent.

Configuration requires `d_model` to be divisible by `n_heads`, `inner_size` to
be odd, and `seq_len` to remain exactly divisible by every successive
`window_size` branching factor so the C-ary pyramid is unambiguous.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/LiuYLLLLD22,
  author       = {Shizhan Liu and
                  Hang Yu and
                  Cong Liao and
                  Jianguo Li and
                  Weiyao Lin and
                  Alex X. Liu and
                  Schahram Dustdar},
  title        = {Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time
                  Series Modeling and Forecasting},
  booktitle    = {The Tenth International Conference on Learning Representations, {ICLR}
                  2022, Virtual Event, April 25-29, 2022},
  publisher    = {OpenReview.net},
  year         = {2022},
  url          = {https://openreview.net/forum?id=0EXmFzUn5I},
  code         = {https://github.com/ant-research/Pyraformer},
  timestamp    = {Fri, 02 Aug 2024 21:59:25 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/LiuYLLLLD22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

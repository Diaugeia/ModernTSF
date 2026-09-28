---
name: "CrossGNN"
summary: "CrossGNN is a multivariate time-series forecasting model that tackles noise and inter-variable heterogeneity through a linear-complexity graph neural network framework. It uses an adaptive multi-scale identifier to build cleaner multi-resolution views of the input, a Cross-Scale GNN to capture trend information at the most informative scale, and a Cross-Variable GNN to jointly model homogeneity and heterogeneity between channels — all while maintaining O(L) time and space complexity with respect to sequence length."
paper: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/9278abf072b58caf21d48dd670b4c721-Abstract-Conference.html"
paper_title: "CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction Refinement"
venue: "NeurIPS 2023"
year: 2023
code: "https://github.com/hqh0728/CrossGNN"
revision: "0407abd085ee8342abe0bbe6de5b2ab17c44373c"
license: "NOASSERTION"
---
# CrossGNN

CrossGNN is a multivariate time-series forecasting model that tackles noise and inter-variable heterogeneity through a linear-complexity graph neural network framework. It uses an adaptive multi-scale identifier to build cleaner multi-resolution views of the input, a Cross-Scale GNN to capture trend information at the most informative scale, and a Cross-Variable GNN to jointly model homogeneity and heterogeneity between channels — all while maintaining O(L) time and space complexity with respect to sequence length.

<!-- model-card:canonical:start -->
## Method overview

CrossGNN is a multivariate time-series forecasting model that tackles noise and inter-variable heterogeneity through a linear-complexity graph neural network framework.

## Core architecture

It uses an adaptive multi-scale identifier to build cleaner multi-resolution views of the input, a Cross-Scale GNN to capture trend information at the most informative scale, and a Cross-Variable GNN to jointly model homogeneity and heterogeneity between channels — all while maintaining O(L) time and space complexity with respect to sequence length.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9278abf072b58caf21d48dd670b4c721-Abstract-Conference.html); title: CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction Refinement; venue/year: NeurIPS 2023 / 2023
- [codebase](https://github.com/hqh0728/CrossGNN); revision: `0407abd085ee8342abe0bbe6de5b2ab17c44373c`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/CrossGNN.toml`](../../../../configs/models/CrossGNN.toml).

## Differences

Pinned source inspection: `models/CrossGNN.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

**Local implementation: confirmed.** The linked repository has no explicit
license and is `reference-only`; its source was inspected at the pinned revision; no external source code was copied. Paper
Eqs. 1--5 map to FFT-selected period pooling; Eqs. 6--10 to scale-sensitive and
trend-preserving temporal edges; Eqs. 11--13 to positive/negative variable
edges; and Eq. 14 to the direct multi-step head. Softplus is used as a smooth
positive relaxation of ReLU graph scores, and data-dependent multiscale length
is interpolated before the fixed-shape horizon head. Dense score construction
does not reproduce the paper's claimed linear-memory implementation. Marks,
external adjacency, checkpoint reference comparison, and published metrics are not supported.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `e_layers=2`, `anti_ood=True`, `tk=3`, `scale_number=4`, `use_tgcn=True`, `use_ngcn=True`, `dropout=0.1`, `tvechidden=8`, `nvechidden=8`, `hidden=16`
<!-- model-card:canonical:end -->

## Paper
- **Title**: CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction Refinement
- **Venue**: NeurIPS 2023
- **Published**: 2023
- **arXiv**: N/A

## Abstract
Recently, multivariate time series (MTS) forecasting techniques have seen rapid development and widespread applications across various fields. Transformer-based and GNN-based methods have shown promising potential due to their strong ability to model interaction of time and variables. However, by conducting a comprehensive analysis of the real-world data, we observe that the temporal fluctuations and heterogeneity between variables are not well handled by existing methods. To address the above issues, we propose CrossGNN, a linear complexity GNN model to refine the cross-scale and cross-variable interaction for MTS. To deal with the unexpected noise in time dimension, an adaptive multi-scale identifier (AMSI) is leveraged to construct multi-scale time series with reduced noise. A Cross-Scale GNN is proposed to extract the scales with clearer trend and weaker noise. Cross-Variable GNN is proposed to utilize the homogeneity and heterogeneity between different variables. By simultaneously focusing on edges with higher saliency scores and constraining those edges with lower scores, the time and space complexity (i.e., O(L)) of CrossGNN can be linear with the input sequence length L. Extensive experimental results on 8 real-world MTS datasets demonstrate the effectiveness of CrossGNN compared with state-of-the-art methods.

## In ModernTSF
Default config: `configs/models/CrossGNN.toml`; model specification: `spec.py`; local implementation: `model.py`.

## Source and verification

Pinned source inspection: `models/CrossGNN.py` were examined at the recorded revision to confirm implementation details. The local module was written for ModernTSF; no external source file is copied.

**Local implementation: confirmed.** The linked repository has no explicit
license and is `reference-only`; its source was inspected at the pinned revision; no external source code was copied. Paper
Eqs. 1--5 map to FFT-selected period pooling; Eqs. 6--10 to scale-sensitive and
trend-preserving temporal edges; Eqs. 11--13 to positive/negative variable
edges; and Eq. 14 to the direct multi-step head. Softplus is used as a smooth
positive relaxation of ReLU graph scores, and data-dependent multiscale length
is interpolated before the fixed-shape horizon head. Dense score construction
does not reproduce the paper's claimed linear-memory implementation. Marks,
external adjacency, checkpoint reference comparison, and published metrics are not supported.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/HuangSZDWZW23,
  author       = {Qihe Huang and
                  Lei Shen and
                  Ruixin Zhang and
                  Shouhong Ding and
                  Binwu Wang and
                  Zhengyang Zhou and
                  Yang Wang},
  editor       = {Alice Oh and
                  Tristan Naumann and
                  Amir Globerson and
                  Kate Saenko and
                  Moritz Hardt and
                  Sergey Levine},
  title        = {CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction
                  Refinement},
  booktitle    = {Advances in Neural Information Processing Systems 36: Annual Conference
                  on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans,
                  LA, USA, December 10 - 16, 2023},
  year         = {2023},
  url          = {http://papers.nips.cc/paper\_files/paper/2023/hash/9278abf072b58caf21d48dd670b4c721-Abstract-Conference.html},
  timestamp    = {Tue, 26 Mar 2024 15:54:05 +0100},
  biburl       = {https://dblp.org/rec/conf/nips/HuangSZDWZW23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

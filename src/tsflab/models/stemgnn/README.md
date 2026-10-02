---
name: "StemGNN"
summary: "StemGNN (Spectral Temporal Graph Neural Network) is a spatiotemporal model for multivariate time-series forecasting that captures inter-series correlations and temporal dependencies jointly in the spectral domain. It combines a Graph Fourier Transform (GFT) for spatial correlation and a Discrete Fourier Transform (DFT) for temporal patterns in a unified end-to-end framework, learning the inter-series graph structure automatically from data without pre-defined priors."
paper: "https://arxiv.org/abs/2103.07719"
paper_title: "Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting"
venue: "NeurIPS 2020"
year: 2020
code: "https://github.com/GestaltCogTeam/BasicTS"
revision: "c218c07b6ce5e4cf908b147fd180c486346fed9c"
license: "Apache-2.0"
---
# StemGNN

StemGNN (Spectral Temporal Graph Neural Network) is a spatiotemporal model for multivariate time-series forecasting that captures inter-series correlations and temporal dependencies jointly in the spectral domain. It combines a Graph Fourier Transform (GFT) for spatial correlation and a Discrete Fourier Transform (DFT) for temporal patterns in a unified end-to-end framework, learning the inter-series graph structure automatically from data without pre-defined priors.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://arxiv.org/abs/2103.07719); title: Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting; venue/year: NeurIPS 2020 / 2020
- [codebase](https://github.com/GestaltCogTeam/BasicTS); revision: `c218c07b6ce5e4cf908b147fd180c486346fed9c`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/StemGNN.toml`](../../../../configs/models/StemGNN.toml).

## Differences

TSFLab rewrites StemGNN locally after reviewing the paper and pinned official codebase. A latent correlation graph drives Chebyshev graph terms and temporal Fourier filtering in doubly residual spectral-temporal blocks. Canonical evidence is stored in [`verification/evidence/StemGNN.json`](../../../../verification/evidence/StemGNN.json).

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `multi_layer=3`, `dropout_rate=0.5`, `leaky_rate=0.2`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting
- **Venue**: NeurIPS 2020
- **Published**: 2020 (arXiv: 2021-03)
- **arXiv**: https://arxiv.org/abs/2103.07719

## Abstract
Multivariate time-series forecasting plays a crucial role in many real-world applications. It is a challenging problem as one needs to consider both intra-series temporal correlations and inter-series correlations simultaneously. Recently, there have been multiple works trying to capture both correlations, but most, if not all of them only capture temporal correlations in the time domain and resort to pre-defined priors as inter-series relationships. In this paper, we propose Spectral Temporal Graph Neural Network (StemGNN) to further improve the accuracy of multivariate time-series forecasting. StemGNN captures inter-series correlations and temporal dependencies jointly in the spectral domain. It combines Graph Fourier Transform (GFT) which models inter-series correlations and Discrete Fourier Transform (DFT) which models temporal dependencies in an end-to-end framework. After passing through GFT and DFT, the spectral representations hold clear patterns and can be predicted effectively by convolution and sequential learning modules. Moreover, StemGNN learns inter-series correlations automatically from the data without using pre-defined priors. We conduct extensive experiments on ten real-world datasets to demonstrate the effectiveness of StemGNN. Code is available at https://github.com/microsoft/StemGNN/

## In TSFLab
Default config: `configs/models/StemGNN.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

TSFLab rewrites StemGNN locally after reviewing the paper and pinned official codebase. A latent correlation graph drives Chebyshev graph terms and temporal Fourier filtering in doubly residual spectral-temporal blocks. Canonical evidence is stored in [`verification/evidence/StemGNN.json`](../../../../verification/evidence/StemGNN.json).

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/CaoWDZZHTXBTZ20,
  author       = {Defu Cao and
                  Yujing Wang and
                  Juanyong Duan and
                  Ce Zhang and
                  Xia Zhu and
                  Congrui Huang and
                  Yunhai Tong and
                  Bixiong Xu and
                  Jing Bai and
                  Jie Tong and
                  Qi Zhang},
  editor       = {Hugo Larochelle and
                  Marc'Aurelio Ranzato and
                  Raia Hadsell and
                  Maria{-}Florina Balcan and
                  Hsuan{-}Tien Lin},
  title        = {Spectral Temporal Graph Neural Network for Multivariate Time-series
                  Forecasting},
  booktitle    = {Advances in Neural Information Processing Systems 33: Annual Conference
                  on Neural Information Processing Systems 2020, NeurIPS 2020, December
                  6-12, 2020, virtual},
  year         = {2020},
  url          = {https://proceedings.neurips.cc/paper/2020/hash/cdf6581cb7aca4b7e19ef136c6e601a5-Abstract.html},
  timestamp    = {Fri, 16 Jan 2026 08:42:14 +0100},
  biburl       = {https://dblp.org/rec/conf/nips/CaoWDZZHTXBTZ20.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

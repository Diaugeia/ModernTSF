---
name: "DCRNN"
summary: "The DCRNN paper combines bidirectional random-walk diffusion convolution with a recurrent encoder-decoder and scheduled sampling for multi-step graph traffic forecasting. This clean-room PyTorch implementation realizes dual random-walk Chebyshev diffusion inside every GRU gate and an autoregressive encoder-decoder without future-target leakage."
paper: "https://openreview.net/forum?id=SJiHXGWAZ"
paper_title: "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting"
venue: "ICLR 2018"
year: 2018
code: "https://github.com/liyaguang/DCRNN"
revision: "602afd9d767d3aa1c9b3eac51710d6aeee12c227"
license: "MIT"
---
# DCRNN

The DCRNN paper combines bidirectional random-walk diffusion convolution with a recurrent encoder-decoder and scheduled sampling for multi-step graph traffic forecasting. This clean-room PyTorch implementation realizes dual random-walk Chebyshev diffusion inside every GRU gate and an autoregressive encoder-decoder without future-target leakage.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://openreview.net/forum?id=SJiHXGWAZ); title: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting; venue/year: ICLR 2018 / 2018
- [codebase](https://github.com/liyaguang/DCRNN); revision: `602afd9d767d3aa1c9b3eac51710d6aeee12c227`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/DCRNN.toml`](../../../../configs/models/DCRNN.toml).

## Differences

- Clean-room implementation: confirmed. The implementation was independently derived from the paper equations; the MIT official repository remains reference-only and no BasicTS implementation code is retained.
- Formula mapping: `DiffusionConvolution` implements the bidirectional random-walk polynomial in Equation (2); `DCGRUCell` applies that operator to reset, update, and candidate equations; the two `RecurrentStack` instances implement the encoder-decoder.
- Adjacency and marks: a validated `adj_mx` supplies forward and reverse transition matrices. Missing adjacency uses identity supports. Raw calendar or node-structured marks are accepted for encoder input features and fitted to `input_dim`.
- Differences and limits: the preset uses one 16-unit layer and three input channels. Scheduled sampling, teacher forcing, official data preprocessing, masked-MAE training, and checkpoint/metric reference comparison are outside this forecast-only implementation.

## Shared components

- [`channel_alignment`](../_components/channel_alignment/README.md)
- [`graph_utils`](../_components/graph_utils/README.md)
- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `rnn_units=16`, `num_rnn_layers=1`, `max_diffusion_step=2`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting
- **Venue**: ICLR 2018
- **Published**: 2018 (arXiv: 2017-07)
- **arXiv**: https://arxiv.org/abs/1707.01926

## Abstract
Spatiotemporal forecasting has various applications in neuroscience, climate and transportation domain. Traffic forecasting is one canonical example of such learning task. The task is challenging due to (1) complex spatial dependency on road networks, (2) non-linear temporal dynamics with changing road conditions and (3) inherent difficulty of long-term forecasting. To address these challenges, we propose to model the traffic flow as a diffusion process on a directed graph and introduce Diffusion Convolutional Recurrent Neural Network (DCRNN), a deep learning framework for traffic forecasting that incorporates both spatial and temporal dependency in the traffic flow. Specifically, DCRNN captures the spatial dependency using bidirectional random walks on the graph, and the temporal dependency using the encoder-decoder architecture with scheduled sampling. We evaluate the framework on two real-world large scale road network traffic datasets and observe consistent improvement of 12% - 15% over state-of-the-art baselines.

## In TSFLab
Default config: `configs/models/DCRNN.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

- Clean-room implementation: confirmed. The implementation was independently derived from the paper equations; the MIT official repository remains reference-only and no BasicTS implementation code is retained.
- Formula mapping: `DiffusionConvolution` implements the bidirectional random-walk polynomial in Equation (2); `DCGRUCell` applies that operator to reset, update, and candidate equations; the two `RecurrentStack` instances implement the encoder-decoder.
- Adjacency and marks: a validated `adj_mx` supplies forward and reverse transition matrices. Missing adjacency uses identity supports. Raw calendar or node-structured marks are accepted for encoder input features and fitted to `input_dim`.
- Differences and limits: the preset uses one 16-unit layer and three input channels. Scheduled sampling, teacher forcing, official data preprocessing, masked-MAE training, and checkpoint/metric reference comparison are outside this forecast-only implementation.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/LiYS018,
  author       = {Yaguang Li and
                  Rose Yu and
                  Cyrus Shahabi and
                  Yan Liu},
  title        = {Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic
                  Forecasting},
  booktitle    = {6th International Conference on Learning Representations, {ICLR} 2018,
                  Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings},
  publisher    = {OpenReview.net},
  year         = {2018},
  url          = {https://openreview.net/forum?id=SJiHXGWAZ},
  timestamp    = {Thu, 25 Jul 2019 14:25:46 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/LiYS018.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

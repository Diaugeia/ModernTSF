---
name: "AirPhyNet"
summary: "AirPhyNet models pollutant diffusion and directed advection as a graph differential equation. This clean-room implementation maps equations (9)-(12) to a GRU posterior, reparameterized initial state, gated diffusion-advection vector field, Euler/RK4 trajectory, and shared decoder."
paper: "https://openreview.net/forum?id=JW3jTjaaAB"
paper_title: "AirPhyNet: Harnessing Physics-Guided Neural Networks for Air Quality Prediction"
venue: "ICLR 2024"
year: 2024
code: "https://github.com/kethmih/AirPhyNet"
revision: "e77576cfea777e8cd07f2ae198c560a8790f4b91"
license: "MIT"
tagline: "Graph neural ODE with gated diffusion-advection vector field, evolved from a sampled GRU initial state."
tags: ["hybrid", "rnn", "gnn", "spatiotemporal", "covariates", "physics-informed", "neural-ode"]
composition: ["normalization=none", "decomposition=none", "temporal=local:euler-rk4-latent-ode-rollout", "channel=local:distance-laplacian-and-directed-flow-operators", "head=local:shared-mlp-decoder", "loss=loss:mse"]
---
# AirPhyNet

## Key ideas

- A per-station GRU encodes pollutant and covariate history into a reparameterized Gaussian initial latent state (sampled only in training).
- `PhysicsVectorField` follows dz/dt = -alpha k tanh(Lz) - (1-alpha) tanh(Mz), with L the distance-graph Laplacian and M a directed flow operator.
- The latent state is integrated one step per horizon with Euler or RK4 and a shared MLP decoder maps each state to concentration.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 24, channels]`. The
declared output contract is a `[batch, 24, channels]` point forecast. Timestamp or exogenous marks are supplied through the runtime batch contract.

## Paper and code

- [paper](https://openreview.net/forum?id=JW3jTjaaAB); title: AirPhyNet: Harnessing Physics-Guided Neural Networks for Air Quality Prediction; venue/year: ICLR 2024 / 2024
- [codebase](https://github.com/kethmih/AirPhyNet); revision: `e77576cfea777e8cd07f2ae198c560a8790f4b91`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/AirPhyNet.toml`](../../../../configs/models/AirPhyNet.toml).

## Differences

- Clean-room implementation: confirmed from the paper; reference source code was not copied.
- Evidence checks the diffusion/advection equation separately, stochastic/evaluation behavior, graph and mark sensitivity, all active gradients, serialization, CPU, and boundaries.

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=24` and `pred_len=24`. Default
model parameters are: `enc_in=8`, `cov_dim=2`, `latent_dim=4`, `rnn_units=64`, `ode_method='rk4'`
<!-- model-card:canonical:end -->

## Paper
- **Title**: AirPhyNet: Harnessing Physics-Guided Neural Networks for Air Quality Prediction
- **Venue**: ICLR 2024
- **Published**: 2024 (arXiv: 2024-02)
- **arXiv**: https://arxiv.org/abs/2402.03784

## Abstract
Air quality prediction and modelling plays a pivotal role in public health and environment management, for individuals and authorities to make informed decisions. Although traditional data-driven models have shown promise in this domain, their long-term prediction accuracy can be limited, especially in scenarios with sparse or incomplete data and they often rely on black-box deep learning structures that lack solid physical foundation leading to reduced transparency and interpretability in predictions. To address these limitations, this paper presents a novel approach named Physics guided Neural Network for Air Quality Prediction (AirPhyNet). Specifically, we leverage two well-established physics principles of air particle movement (diffusion and advection) by representing them as differential equation networks. Then, we utilize a graph structure to integrate physics knowledge into a neural network architecture and exploit latent representations to capture spatio-temporal relationships within the air quality data. Experiments on two real-world benchmark datasets demonstrate that AirPhyNet outperforms state-of-the-art models for different testing scenarios including different lead time (24h, 48h, 72h), sparse data and sudden change prediction, achieving reduction in prediction errors up to 10%. Moreover, a case study further validates that our model captures underlying physical processes of particle movement and generates accurate predictions with real physical meaning.

## In TSFLab
Default config: `configs/models/AirPhyNet.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

Inputs are `x_enc [B, seq_len, N]` and historical raw or node meteorology. Distance and directed flow graphs are construction inputs with explicit ring fallbacks; output is `[B, pred_len, N]`. Equation (9) maps to `encoder/initial_mean/initial_scale`, equations (10)-(11) to `PhysicsVectorField`, and equation (12) to the local differentiable solver and decoder.

## Source and verification

- Clean-room implementation: confirmed from the paper; reference source code was not copied.
- Evidence checks the diffusion/advection equation separately, stochastic/evaluation behavior, graph and mark sensitivity, all active gradients, serialization, CPU, and boundaries.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/HettigeJXLCW24,
  author       = {Kethmi Hirushini Hettige and
                  Jiahao Ji and
                  Shili Xiang and
                  Cheng Long and
                  Gao Cong and
                  Jingyuan Wang},
  title        = {AirPhyNet: Harnessing Physics-Guided Neural Networks for Air Quality
                  Prediction},
  booktitle    = {The Twelfth International Conference on Learning Representations,
                  {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=JW3jTjaaAB},
  timestamp    = {Mon, 13 Jan 2025 16:16:40 +0100},
  biburl       = {https://dblp.org/rec/conf/iclr/HettigeJXLCW24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

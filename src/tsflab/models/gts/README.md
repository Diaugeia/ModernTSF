---
name: "GTS"
summary: "GTS jointly learns a discrete probabilistic graph and a diffusion-recurrent forecaster for multiple time series. This clean-room implementation encodes each node's observed history, classifies every directed edge, samples edges with straight-through Gumbel-Softmax during training, and uses the sampled graph in bidirectional graph-GRU recurrence."
paper: "https://arxiv.org/abs/2101.06861"
paper_title: "Discrete Graph Structure Learning for Forecasting Multiple Time Series"
venue: "ICLR 2021"
year: 2021
code: "https://github.com/GestaltCogTeam/BasicTS"
revision: "c218c07b6ce5e4cf908b147fd180c486346fed9c"
license: "Apache-2.0"
---
# GTS

GTS jointly learns a discrete probabilistic graph and a diffusion-recurrent forecaster for multiple time series. This clean-room implementation encodes each node's observed history, classifies every directed edge, samples edges with straight-through Gumbel-Softmax during training, and uses the sampled graph in bidirectional graph-GRU recurrence.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://arxiv.org/abs/2101.06861); title: Discrete Graph Structure Learning for Forecasting Multiple Time Series; venue/year: ICLR 2021 / 2021
- [codebase](https://github.com/GestaltCogTeam/BasicTS); revision: `c218c07b6ce5e4cf908b147fd180c486346fed9c`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/GTS.toml`](../../../../configs/models/GTS.toml).

## Differences

- Local implementation: the graph learner, diffusion recurrence, and forecasting head are written against TSFLab contracts; BasicTS is retained as a cited reference.
- Formula mapping: `DiscreteGraphDiscovery` implements node-series encoding, pairwise edge probabilities, and differentiable discrete sampling; `LearnedDiffusion` provides bidirectional polynomial graph propagation; `GraphGRUCell` and the encoder/decoder stacks implement the forecasting network.
- Adjacency and marks: supplied adjacency is a shape-checked weak edge-logit prior and the target of `graph_prior_loss`; graph discovery still occurs end-to-end. Encoder marks are accepted through `input_dim`. No future target is consumed.
- Differences and limits: graph features use the current input window rather than a separate full-training-series feature file. Evaluation uses edge probabilities instead of random samples. The auxiliary prior loss, official data pipeline, training schedule, and published metrics remain caller responsibilities.

## Shared components

- [`channel_alignment`](../_components/channel_alignment/README.md)
- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `input_dim=3`, `rnn_units=16`, `num_rnn_layers=1`, `max_diffusion_step=2`, `embedding_dim=16`, `temp=0.5`, `prior_strength=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Discrete Graph Structure Learning for Forecasting Multiple Time Series
- **Venue**: ICLR 2021
- **Published**: 2021 (arXiv: 2021-01)
- **arXiv**: https://arxiv.org/abs/2101.06861

## Abstract
Time series forecasting is an extensively studied subject in statistics, economics, and computer science. Exploration of the correlation and causation among the variables in a multivariate time series shows promise in enhancing the performance of a time series model. When using deep neural networks as forecasting models, we hypothesize that exploiting the pairwise information among multiple (multivariate) time series also improves their forecast. If an explicit graph structure is known, graph neural networks (GNNs) have been demonstrated as powerful tools to exploit the structure. In this work, we propose learning the structure simultaneously with the GNN if the graph is unknown. We cast the problem as learning a probabilistic graph model through optimizing the mean performance over the graph distribution. The distribution is parameterized by a neural network so that discrete graphs can be sampled differentiably through reparameterization. Empirical evaluations show that our method is simpler, more efficient, and better performing than a recently proposed bilevel learning approach for graph structure learning, as well as a broad array of forecasting models, either deep or non-deep learning based, and graph or non-graph based.

## In TSFLab
Default config: `configs/models/GTS.toml`; model specification: `spec.py`; implementation: `model.py`.

## Verification

- Local implementation: the graph learner, diffusion recurrence, and forecasting head are written against TSFLab contracts; BasicTS is retained as a cited reference.
- Formula mapping: `DiscreteGraphDiscovery` implements node-series encoding, pairwise edge probabilities, and differentiable discrete sampling; `LearnedDiffusion` provides bidirectional polynomial graph propagation; `GraphGRUCell` and the encoder/decoder stacks implement the forecasting network.
- Adjacency and marks: supplied adjacency is a shape-checked weak edge-logit prior and the target of `graph_prior_loss`; graph discovery still occurs end-to-end. Encoder marks are accepted through `input_dim`. No future target is consumed.
- Differences and limits: graph features use the current input window rather than a separate full-training-series feature file. Evaluation uses edge probabilities instead of random samples. The auxiliary prior loss, official data pipeline, training schedule, and published metrics remain caller responsibilities.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/Shang0B21,
  author       = {Chao Shang and
                  Jie Chen and
                  Jinbo Bi},
  title        = {Discrete Graph Structure Learning for Forecasting Multiple Time Series},
  booktitle    = {9th International Conference on Learning Representations, {ICLR} 2021,
                  Virtual Event, Austria, May 3-7, 2021},
  publisher    = {OpenReview.net},
  year         = {2021},
  url          = {https://openreview.net/forum?id=WEHSlH5mOk},
  timestamp    = {Wed, 23 Jun 2021 17:36:39 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/Shang0B21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

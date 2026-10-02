---
name: "GCLSTM"
summary: "GCLSTM is a node-level air-quality forecaster that combines spectral graph filtering with LSTM temporal state. This clean-room implementation computes Chebyshev responses jointly for values, covariates, and recurrent state inside all four LSTM gates, then applies a node-wise direct multi-horizon readout."
paper: "https://doi.org/10.1016/j.scitotenv.2019.01.333"
paper_title: "A hybrid model for spatiotemporal forecasting of PM2.5 based on graph convolutional neural network and long short-term memory"
venue: "Science of the Total Environment 2019"
year: 2019
code: "https://github.com/PoorOtterBob/CauAir"
revision: "73dae00ca6ad14abb15174a0a0286d500e868b94"
license: "NOASSERTION"
tagline: "LSTM with all four gates computed by Chebyshev spectral graph convolution of node input and state; node-wise readout."
tags: ["gnn", "rnn", "spatiotemporal", "covariates", "air-quality"]
composition: ["normalization=none", "decomposition=none", "temporal=local:graph-conv-lstm-cell", "channel=component:graph_spectral", "head=local:node-wise-linear-horizon-readout", "loss=loss:mse"]
---
# GCLSTM

## Key ideas

- `chebyshev_supports` (`graph_spectral`) turns the adjacency into scaled-Laplacian Chebyshev supports of order `Ks`.
- `GraphConvLSTMCell` computes input, forget, output and candidate gates jointly from `ChebyshevGraphProjection` of concatenated input and hidden state.
- The cell is unrolled over `seq_len` on value plus calendar-covariate channels (`to_spatiotemporal`, fitted to `1 + cov_dim`).
- A linear layer maps the final node state to all `pred_len` steps.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 24, nodes]`. The
declared output contract is a `[batch, 24, nodes]` point forecast. Timestamp or exogenous marks are supplied through the runtime batch contract.

## Paper and code

- [paper](https://doi.org/10.1016/j.scitotenv.2019.01.333); title: A hybrid model for spatiotemporal forecasting of PM2.5 based on graph convolutional neural network and long short-term memory; venue/year: Science of the Total Environment 2019 / 2019
- [codebase](https://github.com/PoorOtterBob/CauAir); revision: `73dae00ca6ad14abb15174a0a0286d500e868b94`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/GCLSTM.toml`](../../../../configs/models/GCLSTM.toml).

## Differences

- Clean-room implementation: confirmed. No author implementation was identified. The unlicensed CauAir repository is reference-only and its former derived implementation has been deleted; its source was not used for this replacement.
- Formula mapping: `ChebyshevGraphProjection` computes spectral polynomial responses; `GraphConvLSTMCell` produces the input, forget, output, and candidate gates from graph-filtered input and state; `Model.forecast` is the direct node-wise decoder.
- Adjacency and marks: `adj_mx` is shape-checked and converted to scaled-Laplacian Chebyshev supports. Raw calendar stamps and node-structured covariates are accepted and fitted to `1 + cov_dim` channels.
- Differences and limits: the exact paper feature pipeline and graph construction are unavailable. This entry uses one graph-convolutional recurrent cell and direct multi-horizon output rather than reproducing the dataset-specific 72-hour experiment, optimization protocol, or published metrics.

## Shared components

- [`channel_alignment`](../_components/channel_alignment/README.md)
- [`graph_spectral`](../_components/graph_spectral/README.md)
- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=24` and `pred_len=24`. Default
model parameters are: `enc_in=8`, `cov_dim=2`, `Ks=2`, `hidden_dim=64`
<!-- model-card:canonical:end -->

## Paper
- **Title**: A hybrid model for spatiotemporal forecasting of PM2.5 based on graph convolutional neural network and long short-term memory
- **Venue**: Science of the Total Environment, vol. 664, pp. 1-10
- **Published**: 2019
- **arXiv**: N/A

## Abstract
In this paper, we developed a hybrid deep learning approach, which integrates Graph Convolutional networks and Long Short-Term Memory networks (GC-LSTM), to model and forecast the spatiotemporal variation of PM2.5 concentrations. We model historical observations on different stations as spatiotemporal graph series, where air quality variables, meteorological factors, and temporal attributes were used as graph signals. Graph convolutional networks (GCN) were applied to extract the spatial dependency between different stations and LSTM to capture the temporal dependency among observations at different times. The GC-LSTM was trained and tested on real-world data and compared with other state-of-the-art methods. The results showed that GC-LSTM achieved the best performance for predictions with a recall rate of 68.45%, false alarm rate of 4.65% (both at threshold: 115 μg/m³) and correlation coefficient R² of 0.72 for 72-hour forecasts. In addition to PM2.5, the proposed methodology could also be applied to concentration forecasting of different air pollutants in future.

## In TSFLab
Default config: `configs/models/GCLSTM.toml`; model specification: `spec.py`; implementation: `model.py`.

## Verification

- Clean-room implementation: confirmed. No author implementation was identified. The unlicensed CauAir repository is reference-only and its former derived implementation has been deleted; its source was not used for this replacement.
- Formula mapping: `ChebyshevGraphProjection` computes spectral polynomial responses; `GraphConvLSTMCell` produces the input, forget, output, and candidate gates from graph-filtered input and state; `Model.forecast` is the direct node-wise decoder.
- Adjacency and marks: `adj_mx` is shape-checked and converted to scaled-Laplacian Chebyshev supports. Raw calendar stamps and node-structured covariates are accepted and fitted to `1 + cov_dim` channels.
- Differences and limits: the exact paper feature pipeline and graph construction are unavailable. This entry uses one graph-convolutional recurrent cell and direct multi-horizon output rather than reproducing the dataset-specific 72-hour experiment, optimization protocol, or published metrics.

## Citation

```bibtex
@article{Qi2019GCLSTM,
  author    = {Yanlin Qi and
               Qi Li and
               Hamed Karimian and
               Di Liu},
  title     = {A hybrid model for spatiotemporal forecasting of PM2.5 based on graph
               convolutional neural network and long short-term memory},
  journal   = {Science of The Total Environment},
  volume    = {664},
  pages     = {1--10},
  year      = {2019},
  doi       = {10.1016/J.SCITOTENV.2019.01.333}
}
```

---
name: "DFDGCN"
summary: "DFDGCN is a spatiotemporal learning model for node-structured graph data. It captures spatial dependencies in transportation networks by learning dynamic graphs in the frequency domain, mitigating time-shift effects via Fourier transform and combining identity and time embeddings with static predefined and self-adaptive graphs."
paper: "https://doi.org/10.1109/ICASSP48485.2024.10446144"
paper_title: "Dynamic Frequency Domain Graph Convolutional Network for Traffic Forecasting"
venue: "ICASSP 2024"
year: 2024
code: "https://github.com/GestaltCogTeam/DFDGCN"
revision: "3105058512a9279c000e98046a49d1baf3469884"
license: "MIT"
---
# DFDGCN

DFDGCN is a spatiotemporal learning model for node-structured graph data. It captures spatial dependencies in transportation networks by learning dynamic graphs in the frequency domain, mitigating time-shift effects via Fourier transform and combining identity and time embeddings with static predefined and self-adaptive graphs.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://doi.org/10.1109/ICASSP48485.2024.10446144); title: Dynamic Frequency Domain Graph Convolutional Network for Traffic Forecasting; venue/year: ICASSP 2024 / 2024
- [codebase](https://github.com/GestaltCogTeam/DFDGCN); revision: `3105058512a9279c000e98046a49d1baf3469884`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/DFDGCN.toml`](../../../../configs/models/DFDGCN.toml).

## Differences

- Official source: https://github.com/GestaltCogTeam/DFDGCN at `3105058512a9279c000e98046a49d1baf3469884` (MIT).
- Local rewrite: the implementation reconstructs the dilated temporal backbone,
  predefined/adaptive/dynamic graph mixture, frequency-derived graph, node and
  calendar embeddings, and output head using the TSFLab runtime contract.
- Known differences: the default preset uses smaller widths, two blocks instead of the official default four, and top-k 4 for its eight-node contract fixture. Official preprocessing, masked-MAE training, and published numerical results are not included.

## Shared components

- [`adaptive_node_embedding_adjacency`](../_components/adaptive_node_embedding_adjacency/README.md)
- [`gated_dilated_conv`](../_components/gated_dilated_conv/README.md)
- [`graph_utils`](../_components/graph_utils/README.md)
- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `dropout=0.3`, `residual_channels=16`, `dilation_channels=16`, `skip_channels=64`, `end_channels=128`, `kernel_size=2`, `blocks=2`, `layers=2`, `a=1.0`, `fft_emb=10`, `identity_emb=10`, `hidden_emb=30`, `subgraph=4`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Dynamic Frequency Domain Graph Convolutional Network for Traffic Forecasting
- **Venue**: ICASSP 2024
- **Published**: 2024 (arXiv: 2023-12)
- **arXiv**: https://arxiv.org/abs/2312.11933

## Abstract
Complex spatial dependencies in transportation networks make traffic prediction extremely challenging. Much existing work is devoted to learning dynamic graph structures among sensors, and the strategy of mining spatial dependencies from traffic data, known as data-driven, tends to be an intuitive and effective approach. However, Time-Shift of traffic patterns and noise induced by random factors hinder data-driven spatial dependence modeling. In this paper, we propose a novel dynamic frequency domain graph convolution network (DFDGCN) to capture spatial dependencies. Specifically, we mitigate the effects of time-shift by Fourier transform, and introduce the identity embedding of sensors and time embedding when capturing data for graph learning since traffic data with noise is not entirely reliable. The graph is combined with static predefined and self-adaptive graphs during graph convolution to predict future traffic data through classical causal convolutions. Extensive experiments on four real-world datasets demonstrate that our model is effective and outperforms the baselines.

## In TSFLab
Default config: `configs/models/DFDGCN.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

- Official source: https://github.com/GestaltCogTeam/DFDGCN at `3105058512a9279c000e98046a49d1baf3469884` (MIT).
- Local rewrite: the implementation reconstructs the dilated temporal backbone,
  predefined/adaptive/dynamic graph mixture, frequency-derived graph, node and
  calendar embeddings, and output head using the TSFLab runtime contract.
- Known differences: the default preset uses smaller widths, two blocks instead of the official default four, and top-k 4 for its eight-node contract fixture. Official preprocessing, masked-MAE training, and published numerical results are not included.

## Citation

```bibtex
@inproceedings{DBLP:conf/icassp/LiSXQCW24,
  author       = {Yujie Li and
                  Zezhi Shao and
                  Yongjun Xu and
                  Qiang Qiu and
                  Zhaogang Cao and
                  Fei Wang},
  title        = {Dynamic Frequency Domain Graph Convolutional Network for Traffic Forecasting},
  booktitle    = {{IEEE} International Conference on Acoustics, Speech and Signal Processing,
                  {ICASSP} 2024, Seoul, Republic of Korea, April 14-19, 2024},
  pages        = {5245--5249},
  publisher    = {{IEEE}},
  year         = {2024},
  url          = {https://doi.org/10.1109/ICASSP48485.2024.10446144},
  doi          = {10.1109/ICASSP48485.2024.10446144},
  timestamp    = {Sat, 31 May 2025 23:10:02 +0200},
  biburl       = {https://dblp.org/rec/conf/icassp/LiSXQCW24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

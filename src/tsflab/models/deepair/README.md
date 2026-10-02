---
name: "DeepAir"
summary: "DeepAir converts sparse neighboring readings into a consistent target-relative spatial representation before fusing heterogeneous factors. This local implementation retains spatial partition/aggregation/interpolation, five residual HW/WF/SP/MP/HI FusionNets, and the equation (1) horizon-wise weighted sigmoid merge."
paper: "https://doi.org/10.1145/3219819.3219822"
paper_title: "Deep Distributed Fusion Network for Air Quality Prediction"
venue: "KDD 2018"
year: 2018
code: "https://github.com/PoorOtterBob/CauAir"
revision: "73dae00ca6ad14abb15174a0a0286d500e868b94"
license: "NOASSERTION"
tagline: "Target-relative regional aggregation of neighbouring stations feeds five residual fusion nets merged per horizon."
tags: ["mlp", "spatiotemporal", "covariates", "fusion", "air-quality"]
composition: ["normalization=none", "decomposition=none", "temporal=local:residual-fusion-nets-per-factor", "channel=local:spatial-partition-region-aggregation", "head=local:horizon-wise-weighted-sigmoid-merge", "loss=loss:mse"]
---
# DeepAir

## Key ideas

- An explicit `(target, region, source)` projection (`spatial_projection`) turns sparse neighbouring readings into a consistent target-relative regional representation shared as the main feature.
- Five `FusionNet` residual MLP branches handle historical weather, forecast weather, secondary pollutant context, meta properties (calendar and station embedding), and a holistic mix.
- Branch outputs are merged per horizon step and station with learnable softmax weights, then a sigmoid (the paper assumes min-max-scaled targets).
- Future weather covariates come from calendar marks via `marks`.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 24, channels]`. The
declared output contract is a `[batch, 24, channels]` point forecast. Timestamp or exogenous marks are supplied through the runtime batch contract.

## Paper and code

- [paper](https://doi.org/10.1145/3219819.3219822); title: Deep Distributed Fusion Network for Air Quality Prediction; venue/year: KDD 2018 / 2018
- [codebase](https://github.com/PoorOtterBob/CauAir); revision: `73dae00ca6ad14abb15174a0a0286d500e868b94`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/DeepAir.toml`](../../../../configs/models/DeepAir.toml).

## Differences

Pinned source inspection: `src/models/deepair.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

- Local implementation: confirmed from the paper; the unlicensed secondary reference was inspected at the pinned revision; no external source code was copied.
- Evidence verifies spatial aggregation, all five distinct branches, weighted merge, marks/future covariates, graph sensitivity, gradients, serialization, CPU, and boundaries.

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=24` and `pred_len=24`. Default
model parameters are: `enc_in=8`, `cov_dim=2`, `hidden_dim=32`, `spatial_regions=4`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Deep Distributed Fusion Network for Air Quality Prediction
- **Venue**: KDD 2018
- **Published**: 2018
- **arXiv**: N/A

## Abstract
Accompanying the rapid urbanization, many developing countries are suffering from serious air pollution problem. The demand for predicting future air quality is becoming increasingly more important to government's policy-making and people's decision making. In this paper, we predict the air quality of next 48 hours for each monitoring station, considering air quality data, meteorology data, and weather forecast data. Based on the domain knowledge about air pollution, we propose a deep neural network (DNN)-based approach (entitled DeepAir), which consists of a spatial transformation component and a deep distributed fusion network. Considering air pollutants' spatial correlations, the former component converts the spatial sparse air quality data into a consistent input to simulate the pollutant sources. The latter network adopts a neural distributed architecture to fuse heterogeneous urban data for simultaneously capturing the factors affecting air quality, e.g. meteorological conditions. We deployed DeepAir in our AirPollutionPrediction system, providing fine-grained air quality forecasts for 300+ Chinese cities every hour. The experimental results on the data from three-year nine Chinese-city demonstrate the advantages of DeepAir beyond 10 baseline methods. Comparing with the previous online approach in AirPollutionPrediction system, we have 2.4%, 12.2%, 63.2% relative accuracy improvements on short-term, long-term and sudden changes prediction, respectively.

## In TSFLab
Default config: `configs/models/DeepAir.toml`; model specification: `spec.py`; local implementation: `model.py`.

Inputs are `x_enc [B, seq_len, N]`, historical/future node covariates, and `spatial_mx [N, regions, N]`; a 2-D adjacency is accepted as one region. The normalized output is `[B, pred_len, N]` in `(0,1)`. Regional target history supplies the unavailable secondary-pollutant branch in the generic contract, while exact coordinates, pollutant panels, terrain, and min-max statistics remain dataset responsibilities.

## Source and verification

Pinned source inspection: `src/models/deepair.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

- Local implementation: confirmed from the paper; the unlicensed secondary reference was inspected at the pinned revision; no external source code was copied.
- Evidence verifies spatial aggregation, all five distinct branches, weighted merge, marks/future covariates, graph sensitivity, gradients, serialization, CPU, and boundaries.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/YiZWLZ18,
  author       = {Xiuwen Yi and
                  Junbo Zhang and
                  Zhaoyuan Wang and
                  Tianrui Li and
                  Yu Zheng},
  editor       = {Yike Guo and
                  Faisal Farooq},
  title        = {Deep Distributed Fusion Network for Air Quality Prediction},
  booktitle    = {Proceedings of the 24th {ACM} {SIGKDD} International Conference on
                  Knowledge Discovery {\&} Data Mining, {KDD} 2018, London, UK, August
                  19-23, 2018},
  pages        = {965--973},
  publisher    = {{ACM}},
  year         = {2018},
  url          = {https://doi.org/10.1145/3219819.3219822},
  doi          = {10.1145/3219819.3219822},
  timestamp    = {Sun, 02 Nov 2025 21:27:16 +0100},
  biburl       = {https://dblp.org/rec/conf/kdd/YiZWLZ18.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

---
name: "AirFormer"
summary: "AirFormer factorizes nationwide air-quality learning into deterministic temporal/spatial stages and a stochastic uncertainty stage. This local implementation includes growing-window CT-MSA, explicit query-relative dartboard DS-MSA, residual blocks, and a reverse-level Gaussian latent hierarchy."
paper: "https://doi.org/10.1609/aaai.v37i12.26676"
paper_title: "AirFormer: Predicting Nationwide Air Quality in China with Transformers"
venue: "AAAI 2023"
year: 2023
code: "https://github.com/yoshall/airformer"
revision: "ef7d3933768490e3a06921b8eb0f837c61741194"
license: "NOASSERTION"
---
# AirFormer

AirFormer factorizes nationwide air-quality learning into deterministic causal-temporal and dartboard-spatial attention, followed by a top-down latent hierarchy for uncertainty. This implementation is independently derived from the AAAI paper.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 24, channels]`. The
declared output contract is a `[batch, 24, channels]` point forecast. Timestamp or exogenous marks are supplied through the runtime batch contract.

## Paper and code

- [paper](https://doi.org/10.1609/aaai.v37i12.26676); title: AirFormer: Predicting Nationwide Air Quality in China with Transformers; venue/year: AAAI 2023 / 2023
- [codebase](https://github.com/yoshall/airformer); revision: `ef7d3933768490e3a06921b8eb0f837c61741194`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/AirFormer.toml`](../../../../configs/models/AirFormer.toml).

## Differences

Pinned source inspection: `src/tsflab/models/airformer.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

- Local implementation: confirmed from the paper; reference-only source code was inspected at the pinned revision; no external source code was copied.
- Evidence covers causal windows, dartboard aggregation, stochastic/evaluation behavior, graph and mark sensitivity, complete gradients, serialization, CPU, and boundaries.

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=24` and `pred_len=24`. Default
model parameters are: `enc_in=8`, `cov_dim=2`, `d_model=32`, `nhead=2`, `num_encoder_layers=4`, `spatial_regions=4`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: AirFormer: Predicting Nationwide Air Quality in China with Transformers
- **Venue**: AAAI 2023
- **Published**: 2023 (arXiv: 2022-11)
- **arXiv**: https://arxiv.org/abs/2211.15979

## Abstract
Air pollution is a crucial issue affecting human health and livelihoods, as well as one of the barriers to economic and social growth. Forecasting air quality has become an increasingly important endeavor with significant social impacts, especially in emerging countries like China. In this paper, we present a novel Transformer architecture termed AirFormer to collectively predict nationwide air quality in China, with an unprecedented fine spatial granularity covering thousands of locations. AirFormer decouples the learning process into two stages -- 1) a bottom-up deterministic stage that contains two new types of self-attention mechanisms to efficiently learn spatio-temporal representations; 2) a top-down stochastic stage with latent variables to capture the intrinsic uncertainty of air quality data. We evaluate AirFormer with 4-year data from 1,085 stations in the Chinese Mainland. Compared to the state-of-the-art model, AirFormer reduces prediction errors by 5%~8% on 72-hour future predictions. Our source code is available at this https URL.

## In TSFLab
Default config: `configs/models/AirFormer.toml`; model specification: `spec.py`; local implementation: `model.py`.

Inputs are `x_enc [B, seq_len, N]` and raw or node-structured covariates. Exact spatial preprocessing is `dartboard_mx [N, M, N]`; an ordinary adjacency is treated as one region, while the fallback is explicitly topological rather than geographic. CT-MSA is causal, DS-MSA attends from every station to its regional aggregates, training samples each top-down latent, evaluation uses latent means, and the public output is `[B, pred_len, N]`.

## Source and verification

Pinned source inspection: `src/tsflab/models/airformer.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

- Local implementation: confirmed from the paper; reference-only source code was inspected at the pinned revision; no external source code was copied.
- Evidence covers causal windows, dartboard aggregation, stochastic/evaluation behavior, graph and mark sensitivity, complete gradients, serialization, CPU, and boundaries.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/LiangXKWWZZZ23,
  author       = {Yuxuan Liang and
                  Yutong Xia and
                  Songyu Ke and
                  Yiwei Wang and
                  Qingsong Wen and
                  Junbo Zhang and
                  Yu Zheng and
                  Roger Zimmermann},
  editor       = {Brian Williams and
                  Yiling Chen and
                  Jennifer Neville},
  title        = {AirFormer: Predicting Nationwide Air Quality in China with Transformers},
  booktitle    = {Thirty-Seventh {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2023, Thirty-Fifth Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2023, Thirteenth Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2023, Washington, DC, USA, February
                  7-14, 2023},
  pages        = {14329--14337},
  publisher    = {{AAAI} Press},
  year         = {2023},
  url          = {https://doi.org/10.1609/aaai.v37i12.26676},
  doi          = {10.1609/AAAI.V37I12.26676},
  timestamp    = {Wed, 18 Mar 2026 17:07:12 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/LiangXKWWZZZ23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

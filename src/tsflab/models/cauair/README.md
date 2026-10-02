---
name: "CauAir"
summary: "CauAir models nationwide AQI-weather causality through learnable coarse-region caches with linear station complexity. This local implementation retains multi-head cache assignment/aggregation, equation (5) parallel CachLormer with SwiGLU, past association, future-covariate propagation, and horizon decoding."
paper: "https://www.ijcai.org/proceedings/2025/353"
paper_title: "Causal Learning Meet Covariates: Empowering Lightweight and Effective Nationwide Air Quality Forecasting"
venue: "IJCAI 2025"
year: 2025
code: "https://github.com/PoorOtterBob/CauAir"
revision: "73dae00ca6ad14abb15174a0a0286d500e868b94"
license: "NOASSERTION"
---
# CauAir

CauAir uses learnable coarse-region caches to model causal association between station AQI and weather at linear complexity in the number of stations. This entry is a local implementation from the IJCAI paper; the unlicensed author repository is reference-only and was inspected at the pinned revision; no external source code was copied.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 24, channels]`. The
declared output contract is a `[batch, 24, channels]` point forecast. Timestamp or exogenous marks are supplied through the runtime batch contract.

## Paper and code

- [paper](https://www.ijcai.org/proceedings/2025/353); title: Causal Learning Meet Covariates: Empowering Lightweight and Effective Nationwide Air Quality Forecasting; venue/year: IJCAI 2025 / 2025
- [codebase](https://github.com/PoorOtterBob/CauAir); revision: `73dae00ca6ad14abb15174a0a0286d500e868b94`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/CauAir.toml`](../../../../configs/models/CauAir.toml).

## Differences

Pinned source inspection: `src/tsflab/models/cauair.py` was examined at the recorded
revision to confirm implementation details. The local module was written for
TSFLab; no external source file is copied.

- Local implementation: confirmed; reference-only source code was inspected at the pinned revision; no external source code was copied.
- Evidence covers cache assignment/complexity, the parallel CachLormer equation, historical/future covariate effects, all active gradients, serialization, CPU, and boundaries.

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=24` and `pred_len=24`. Default
model parameters are: `enc_in=6`, `dim=64`, `cache_count=8`, `heads=4`, `cov_dim=2`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Causal Learning Meet Covariates: Empowering Lightweight and Effective Nationwide Air Quality Forecasting
- **Venue**: IJCAI 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
Air quality prediction plays a crucial role in the development of smart cities, garnering significant attention from both academia and industry. Current air quality prediction models encounter two major limitations: their high computational complexity limits scalability to nationwide datasets, and they often regard weather covariates as optional auxiliary information. In reality, weather covariates can have a substantial impact on air quality indices (AQI), exhibiting a significant causal association. In this paper, we first present a nationwide air quality dataset to address the lack of open-source, large-scale datasets in this field. Then we propose a causal learning model, CauAir, for air quality prediction that harnesses the powerful representation capabilities of the Transformer to explicitly model the causal association between weather covariates and AQI. To address the high complexity of traditional Transformers, we design CachLormer, which features two key innovations: a simplified architecture with redundant components removed, and a cache-attention mechanism that employs learnable embeddings for perceiving causal association between AQI and weather covariates in a coarse-grained perspective. We use information theory to illustrate the superiority of the proposed model. Finally, experimental results on three datasets with 28 as the baseline demonstrate that our model achieves competitive performance, while maintaining high training efficiency and low memory consumption.

## In TSFLab
Default config: `configs/models/CauAir.toml`; model specification: `spec.py`; local implementation: `model.py`.

Inputs are `x_enc [B, seq_len, N]` plus historical and future weather `[B, time, N, cov_dim]`; raw timestamps provide a two-calendar-feature fallback. Output is `[B, pred_len, N]`, and adjacency is deliberately ignored because cache-attention is graph-free. Cache assignment and reconstruction map to equations (6)-(11), while the two causal stages remain explicit in `Model.forward`.

## Verification

Pinned source inspection: `src/tsflab/models/cauair.py` was examined at the recorded
revision to confirm implementation details. The local module was written for
TSFLab; no external source file is copied.

- Local implementation: confirmed; reference-only source code was inspected at the pinned revision; no external source code was copied.
- Evidence covers cache assignment/complexity, the parallel CachLormer equation, historical/future covariate effects, all active gradients, serialization, CPU, and boundaries.

## Citation

```bibtex
@inproceedings{DBLP:conf/ijcai/MaCW0ZZW25,
  author       = {Jiaming Ma and
                  Zhiqing Cui and
                  Binwu Wang and
                  Pengkun Wang and
                  Zhengyang Zhou and
                  Zhe Zhao and
                  Yang Wang},
  title        = {Causal Learning Meet Covariates: Empowering Lightweight and Effective
                  Nationwide Air Quality Forecasting},
  booktitle    = {Proceedings of the Thirty-Fourth International Joint Conference on
                  Artificial Intelligence, {IJCAI} 2025, Montreal, Canada, August 16-22,
                  2025},
  pages        = {3171--3179},
  publisher    = {ijcai.org},
  year         = {2025},
  url          = {https://doi.org/10.24963/ijcai.2025/353},
  doi          = {10.24963/IJCAI.2025/353},
  timestamp    = {Wed, 24 Sep 2025 17:45:28 +0200},
  biburl       = {https://dblp.org/rec/conf/ijcai/MaCW0ZZW25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

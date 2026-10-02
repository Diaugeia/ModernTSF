---
name: "RPMixer"
summary: "RPMixer is a spatiotemporal forecasting model built on an all-MLP (all-Multi-Layer Perceptron) architecture that forgoes explicit graph-based spatial modeling in favour of general time series mixing. It addresses the tendency of standard MLP-mixer models to overfit on large-scale spatial-temporal datasets by inserting random projection layers between blocks to increase output diversity, exploiting the ensemble-like behaviour of deep residual networks where each block acts as a base learner. The approach achieves competitive or superior performance against both graph-based and general forecasting baselines on large spatial-temporal benchmarks."
paper: "https://doi.org/10.1145/3637528.3671881"
paper_title: "RPMixer: Shaking Up Time Series Forecasting with Random Projections for Large Spatial-Temporal Data"
venue: "KDD 2024"
year: 2024
code: "https://github.com/PoorOtterBob/CauAir"
revision: "73dae00ca6ad14abb15174a0a0286d500e868b94"
license: "NOASSERTION"
---
# RPMixer

RPMixer is a spatiotemporal forecasting model built on an all-MLP (all-Multi-Layer Perceptron) architecture that forgoes explicit graph-based spatial modeling in favour of general time series mixing. It addresses the tendency of standard MLP-mixer models to overfit on large-scale spatial-temporal datasets by inserting random projection layers between blocks to increase output diversity, exploiting the ensemble-like behaviour of deep residual networks where each block acts as a base learner. The approach achieves competitive or superior performance against both graph-based and general forecasting baselines on large spatial-temporal benchmarks.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://doi.org/10.1145/3637528.3671881); title: RPMixer: Shaking Up Time Series Forecasting with Random Projections for Large Spatial-Temporal Data; venue/year: KDD 2024 / 2024
- [codebase](https://github.com/PoorOtterBob/CauAir); revision: `73dae00ca6ad14abb15174a0a0286d500e868b94`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/RPMixer.toml`](../../../../configs/models/RPMixer.toml).

## Differences

Pinned source inspection: `src/moderntsf/models/rpmixer.py` was examined at the recorded
revision to confirm implementation details. The local module was written for
ModernTSF; no external source file is copied.

Local implementation: confirmed.

Local implementation confirmed from paper equations (1)--(6); the unlicensed CauAir reference was inspected at the pinned revision; no external source code was copied, and its former `_official reference.py` derivative has been removed. Each block uses pre-activation complex FFT-domain temporal projection, a distinct frozen random node projection, learned spatial reconstruction, and exact identity residual paths before a shared horizon decoder. The local runtime uses only observed node values, deliberately ignores graph adjacency and timestamp marks as the paper's graph-free formulation permits, and does not reproduce extra feature construction, MAE training, or benchmark hyperparameters.

Component audit (`mixer_block`): `RPMixerBlock` mixes through a complex FFT-domain linear map and a fixed random node projection with no `LayerNorm` at all (Eqs. 1--6), unlike `mixer_block`'s pre-normalized time/feature MLPs; it stays model-local. The horizon decoder already reuses the cataloged `channel_wise_linear`.

## Shared components

- [`channel_wise_linear`](../_components/channel_wise_linear/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`, `random_dim=4`, `e_layers=3`
<!-- model-card:canonical:end -->

## Paper
- **Title**: RPMixer: Shaking Up Time Series Forecasting with Random Projections for Large Spatial-Temporal Data
- **Venue**: KDD 2024
- **Published**: 2024 (arXiv: 2024-02)
- **arXiv**: https://arxiv.org/abs/2402.10487

## Abstract
Spatial-temporal forecasting systems play a crucial role in addressing numerous real-world challenges. In this paper, we investigate the potential of addressing spatial-temporal forecasting problems using general time series forecasting models, i.e., models that do not leverage the spatial relationships among the nodes. We propose a all-Multi-Layer Perceptron (all-MLP) time series forecasting architecture called RPMixer. The all-MLP architecture was chosen due to its recent success in time series forecasting benchmarks. Furthermore, our method capitalizes on the ensemble-like behavior of deep neural networks, where each individual block within the network behaves like a base learner in an ensemble model, particularly when identity mapping residual connections are incorporated. By integrating random projection layers into our model, we increase the diversity among the blocks' outputs, thereby improving the overall performance of the network. Extensive experiments conducted on the largest spatial-temporal forecasting benchmark datasets demonstrate that the proposed method outperforms alternative methods, including both spatial-temporal graph models and general forecasting models.

## In ModernTSF
Default config: `configs/models/RPMixer.toml`; model specification: `spec.py`; implementation: `model.py`.

## Verification

Pinned source inspection: `src/moderntsf/models/rpmixer.py` was examined at the recorded
revision to confirm implementation details. The local module was written for
ModernTSF; no external source file is copied.

Local implementation: confirmed.

Local implementation confirmed from paper equations (1)--(6); the unlicensed CauAir reference was inspected at the pinned revision; no external source code was copied, and its former `_official reference.py` derivative has been removed. Each block uses pre-activation complex FFT-domain temporal projection, a distinct frozen random node projection, learned spatial reconstruction, and exact identity residual paths before a shared horizon decoder. The local runtime uses only observed node values, deliberately ignores graph adjacency and timestamp marks as the paper's graph-free formulation permits, and does not reproduce extra feature construction, MAE training, or benchmark hyperparameters.

Component audit (`mixer_block`): `RPMixerBlock` mixes through a complex FFT-domain linear map and a fixed random node projection with no `LayerNorm` at all (Eqs. 1--6), unlike `mixer_block`'s pre-normalized time/feature MLPs; it stays model-local. The horizon decoder already reuses the cataloged `channel_wise_linear`.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/YehFDSLAWC0ZWZ24,
  author       = {Chin{-}Chia Michael Yeh and
                  Yujie Fan and
                  Xin Dai and
                  Uday Singh Saini and
                  Vivian Lai and
                  Prince Osei Aboagye and
                  Junpeng Wang and
                  Huiyuan Chen and
                  Yan Zheng and
                  Zhongfang Zhuang and
                  Liang Wang and
                  Wei Zhang},
  editor       = {Ricardo Baeza{-}Yates and
                  Francesco Bonchi},
  title        = {RPMixer: Shaking Up Time Series Forecasting with Random Projections
                  for Large Spatial-Temporal Data},
  booktitle    = {Proceedings of the 30th {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, {KDD} 2024, Barcelona, Spain, August 25-29, 2024},
  pages        = {3919--3930},
  publisher    = {{ACM}},
  year         = {2024},
  url          = {https://doi.org/10.1145/3637528.3671881},
  doi          = {10.1145/3637528.3671881},
  timestamp    = {Mon, 24 Nov 2025 11:08:37 +0100},
  biburl       = {https://dblp.org/rec/conf/kdd/YehFDSLAWC0ZWZ24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

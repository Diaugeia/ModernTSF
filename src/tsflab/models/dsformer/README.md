---
name: "DSFormer"
summary: "DSFormer (Double Sampling Transformer) is a Transformer-based model for multivariate long-term time series forecasting. It combines a Double Sampling (DS) block — which applies down-sampling and piecewise sampling to capture global and local temporal information — with a Temporal Variable Attention (TVA) block that mines both temporal and inter-variable dependencies, feeding a generative MLP decoder to produce multi-horizon forecasts."
paper: "https://arxiv.org/abs/2308.03274"
paper_title: "DSformer: A Double Sampling Transformer for Multivariate Time Series Long-term Prediction"
venue: "CIKM 2023"
year: 2023
code: "https://github.com/GestaltCogTeam/DSformer"
revision: "ccdbc354603e7842a89603649b0e33a8142c7701"
license: "NOASSERTION"
tagline: "Double sampling (piecewise and interval views) with temporal-variable attention, gated fusion, channel decoder."
tags: ["transformer", "channel-mixing", "attention-variant", "multi-scale", "normalization"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=local:double-sampling-views+local:temporal-attention", "channel=local:variable-attention-gated-cross-fusion", "head=local:decoder-attention-linear-head", "loss=loss:mse"]
---
# DSFormer

## Key ideas

- `dual_sampling` reshapes each series into a piecewise view (contiguous segments, local information) and an interval view (strided samples, global information).
- `TVABlock` runs temporal attention within each channel and variable attention across channels, then blends them with a sigmoid cross gate and a feed-forward layer.
- The two views are processed by separate block stacks, concatenated and mixed linearly (`node_mix`), refined by `decoder_attention`, and mapped to the horizon; `revin` wraps the model.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2308.03274); title: DSformer: A Double Sampling Transformer for Multivariate Time Series Long-term Prediction; venue/year: CIKM 2023 / 2023
- [codebase](https://github.com/GestaltCogTeam/DSformer); revision: `ccdbc354603e7842a89603649b0e33a8142c7701`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/DSFormer.toml`](../../../../configs/models/DSFormer.toml).

## Differences

Pinned source inspection: `main_model.py`, `block/TVA_block.py`, `block/decoder_block.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

Clean-room structure map: dual sampling; temporal attention; variable attention; gated cross-view fusion; channel decoder; RevIN restoration. The reference code has no declared license and was inspected at the pinned revision; no external source code was copied.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `num_layer=1`, `muti_head=2`, `num_samp=2`, `dropout=0.15`, `if_node=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: DSformer: A Double Sampling Transformer for Multivariate Time Series Long-term Prediction
- **Venue**: CIKM 2023
- **Published**: 2023 (arXiv: 2023-08)
- **arXiv**: https://arxiv.org/abs/2308.03274

## Abstract
Multivariate time series long-term prediction, which aims to predict the change of data in a long time, can provide references for decision-making. Although transformer-based models have made progress in this field, they usually do not make full use of three features of multivariate time series: global information, local information, and variables correlation. To effectively mine the above three features and establish a high-precision prediction model, we propose a double sampling transformer (DSformer), which consists of the double sampling (DS) block and the temporal variable attention (TVA) block. Firstly, the DS block employs down sampling and piecewise sampling to transform the original series into feature vectors that focus on global information and local information respectively. Then, TVA block uses temporal attention and variable attention to mine these feature vectors from different dimensions and extract key information. Finally, based on a parallel structure, DSformer uses multiple TVA blocks to mine and integrate different features obtained from DS blocks respectively. The integrated feature information is passed to the generative decoder based on a multi-layer perceptron to realize multivariate time series long-term prediction. Experimental results on nine real-world datasets show that DSformer can outperform eight existing baselines.

## In TSFLab
Default config: `configs/models/DSFormer.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Pinned source inspection: `main_model.py`, `block/TVA_block.py`, `block/decoder_block.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

Clean-room structure map: dual sampling; temporal attention; variable attention; gated cross-view fusion; channel decoder; RevIN restoration. The reference code has no declared license and was inspected at the pinned revision; no external source code was copied.

## Citation

```bibtex
@inproceedings{DBLP:conf/cikm/YuWSSWX23,
  author       = {Chengqing Yu and
                  Fei Wang and
                  Zezhi Shao and
                  Tao Sun and
                  Lin Wu and
                  Yongjun Xu},
  editor       = {Ingo Frommholz and
                  Frank Hopfgartner and
                  Mark Lee and
                  Michael Oakes and
                  Mounia Lalmas and
                  Min Zhang and
                  Rodrygo L. T. Santos},
  title        = {DSformer: {A} Double Sampling Transformer for Multivariate Time Series
                  Long-term Prediction},
  booktitle    = {Proceedings of the 32nd {ACM} International Conference on Information
                  and Knowledge Management, {CIKM} 2023, Birmingham, United Kingdom,
                  October 21-25, 2023},
  pages        = {3062--3072},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3583780.3614851},
  doi          = {10.1145/3583780.3614851},
  timestamp    = {Mon, 10 Feb 2025 16:22:03 +0100},
  biburl       = {https://dblp.org/rec/conf/cikm/YuWSSWX23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

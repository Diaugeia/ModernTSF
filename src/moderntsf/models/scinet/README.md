---
name: "SCINet"
summary: "SCINet is a CNN-based time-series forecasting model that exploits the property that temporal relations are largely preserved after downsampling. It uses a recursive downsample-convolve-interact architecture in which each layer splits the sequence into two sub-sequences, applies distinct convolutional filters to each, and then fuses them — enabling multi-resolution temporal feature extraction at O(L) complexity."
paper: "https://arxiv.org/abs/2106.09305"
paper_title: "SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction"
venue: "NeurIPS 2022"
year: 2022
code: "https://github.com/thuml/Time-Series-Library"
revision: "4e938a1767106324dd753b2a44832bf870a0252e"
license: "MIT"
---
# SCINet

SCINet is a CNN-based time-series forecasting model that exploits the property that temporal relations are largely preserved after downsampling. It uses a recursive downsample-convolve-interact architecture in which each layer splits the sequence into two sub-sequences, applies distinct convolutional filters to each, and then fuses them — enabling multi-resolution temporal feature extraction at O(L) complexity.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2106.09305); title: SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction; venue/year: NeurIPS 2022 / 2022
- [codebase](https://github.com/thuml/Time-Series-Library); revision: `4e938a1767106324dd753b2a44832bf870a0252e`; license: `MIT`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/SCINet.toml`](../../../../configs/models/SCINet.toml).

## Differences

**Clean-room implementation: confirmed.** The published interaction equations, recursive node count, interleaving, stacked forecast path, gradients, and runtime boundaries have executable tests. Inputs are `[B, seq_len, enc_in]`, outputs are `[B, pred_len, enc_in]`, and marks are ignored. Intermediate supervision remains a trainer concern. The linked implementation is reference-only and no code was copied.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `num_stacks=1`, `num_levels=3`, `kernel_size=5`, `dropout=0.0`
<!-- model-card:canonical:end -->

## Paper
- **Title**: SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction
- **Venue**: NeurIPS 2022
- **Published**: 2022 (arXiv: 2021-06)
- **arXiv**: https://arxiv.org/abs/2106.09305

## Abstract
One unique property of time series is that the temporal relations are largely preserved after downsampling into two sub-sequences. By taking advantage of this property, we propose a novel neural network architecture that conducts sample convolution and interaction for temporal modeling and forecasting, named SCINet. Specifically, SCINet is a recursive downsample-convolve-interact architecture. In each layer, we use multiple convolutional filters to extract distinct yet valuable temporal features from the downsampled sub-sequences or features. By combining these rich features aggregated from multiple resolutions, SCINet effectively models time series with complex temporal dynamics. Experimental results show that SCINet achieves significant forecasting accuracy improvements over both existing convolutional models and Transformer-based solutions across various real-world time series forecasting datasets. Our codes and data are available at https://github.com/cure-lab/SCINet.

## In ModernTSF
Default config: `configs/models/SCINet.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

**Clean-room implementation: confirmed.** The published interaction equations, recursive node count, interleaving, stacked forecast path, gradients, and runtime boundaries have executable tests. Inputs are `[B, seq_len, enc_in]`, outputs are `[B, pred_len, enc_in]`, and marks are ignored. Intermediate supervision remains a trainer concern. The linked implementation is reference-only and no code was copied.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/LiuZCXLM022,
  author       = {Minhao Liu and
                  Ailing Zeng and
                  Muxi Chen and
                  Zhijian Xu and
                  Qiuxia Lai and
                  Lingna Ma and
                  Qiang Xu},
  editor       = {Sanmi Koyejo and
                  S. Mohamed and
                  A. Agarwal and
                  Danielle Belgrave and
                  K. Cho and
                  A. Oh},
  title        = {SCINet: Time Series Modeling and Forecasting with Sample Convolution
                  and Interaction},
  booktitle    = {Advances in Neural Information Processing Systems 35: Annual Conference
                  on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans,
                  LA, USA, November 28 - December 9, 2022},
  year         = {2022},
  url          = {http://papers.nips.cc/paper\_files/paper/2022/hash/266983d0949aed78a16fa4782237dea7-Abstract-Conference.html},
  timestamp    = {Mon, 08 Jan 2024 16:31:36 +0100},
  biburl       = {https://dblp.org/rec/conf/nips/LiuZCXLM022.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

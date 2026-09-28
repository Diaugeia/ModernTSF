---
name: "HDMixer"
summary: "HDMixer is a pure MLP-based time series forecasting model for multivariate prediction that addresses two limitations of standard patch-based approaches: fixed-length patches lose temporal boundary information (e.g., peaks and periods are cut arbitrarily), and existing methods focus mainly on long-range cross-patch dependencies while ignoring short-range within-patch and cross-variable interactions. HDMixer introduces a Length-Extendable Patcher (LEP) to enrich patch boundary information and a Hierarchical Dependency Explorer (HDE) that models all three dependency levels — within-patch (short-term), across-patch (long-term), and cross-variable — using pure MLPs."
paper: "https://ojs.aaai.org/index.php/AAAI/article/view/29155"
paper_title: "HDMixer: Hierarchical Dependency with Extendable Patch for Multivariate Time Series Forecasting"
venue: "AAAI 2024"
year: 2024
code: "https://github.com/hqh0728/HDMixer"
revision: "da17f94b63b869633556b6bf65a5c68e3f322e2b"
license: "NOASSERTION"
---
# HDMixer

HDMixer is a pure MLP-based time series forecasting model for multivariate prediction that addresses two limitations of standard patch-based approaches: fixed-length patches lose temporal boundary information (e.g., peaks and periods are cut arbitrarily), and existing methods focus mainly on long-range cross-patch dependencies while ignoring short-range within-patch and cross-variable interactions. HDMixer introduces a Length-Extendable Patcher (LEP) to enrich patch boundary information and a Hierarchical Dependency Explorer (HDE) that models all three dependency levels — within-patch (short-term), across-patch (long-term), and cross-variable — using pure MLPs.

<!-- model-card:canonical:start -->
## Method overview

HDMixer is a pure MLP-based time series forecasting model for multivariate prediction that addresses two limitations of standard patch-based approaches: fixed-length patches lose temporal boundary information (e.g., peaks and periods are cut arbitrarily), and existing methods focus mainly on long-range cross-patch dependencies while ignoring short-range within-patch and cross-variable interactions.

## Core architecture

HDMixer introduces a Length-Extendable Patcher (LEP) to enrich patch boundary information and a Hierarchical Dependency Explorer (HDE) that models all three dependency levels — within-patch (short-term), across-patch (long-term), and cross-variable — using pure MLPs.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://ojs.aaai.org/index.php/AAAI/article/view/29155); title: HDMixer: Hierarchical Dependency with Extendable Patch for Multivariate Time Series Forecasting; venue/year: AAAI 2024 / 2024
- [codebase](https://github.com/hqh0728/HDMixer); revision: `da17f94b63b869633556b6bf65a5c68e3f322e2b`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/HDMixer.toml`](../../../../configs/models/HDMixer.toml).

## Differences

Clean-room implementation: confirmed. LEP and HDE were independently designed from the paper description; reference-only repository source was not copied. The explicit center/width sampler is a disclosed local choice.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `d_ff=256`, `e_layers=3`, `patch_len=16`, `stride=8`, `dropout=0.1`, `head_dropout=0.0`, `revin=True`, `affine=True`, `subtract_last=False`, `deform_range=0.25`, `mix_time=True`, `mix_variable=True`, `mix_channel=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: HDMixer: Hierarchical Dependency with Extendable Patch for Multivariate Time Series Forecasting
- **Venue**: AAAI 2024
- **Published**: 2024
- **arXiv**: N/A

## Abstract
Multivariate time series (MTS) prediction has been widely adopted in various scenarios. Recently, some methods have employed patching to enhance local semantics and improve model performance. However, length-fixed patch are prone to losing temporal boundary information, such as complete peaks and periods. Moreover, existing methods mainly focus on modeling long-term dependencies across patches, while paying little attention to other dimensions (e.g., short-term dependencies within patches and complex interactions among cross-variavle patches). To address these challenges, we propose a pure MLP-based HDMixer, aiming to acquire patches with richer semantic information and efficiently modeling hierarchical interactions. Specifically, we design a Length-Extendable Patcher (LEP) tailored to MTS, which enriches the boundary information of patches and alleviates semantic incoherence in series. Subsequently, we devise a Hierarchical Dependency Explorer (HDE) based on pure MLPs. This explorer effectively models short-term dependencies within patches, long-term dependencies across patches, and complex interactions among variables. Extensive experiments on 9 real-world datasets demonstrate the superiority of our approach.

## In ModernTSF
Default config: `configs/models/HDMixer.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed. LEP and HDE were independently designed from the paper description; reference-only repository source was not copied. The explicit center/width sampler is a disclosed local choice.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/HuangSZCDZW24,
  author       = {Qihe Huang and
                  Lei Shen and
                  Ruixin Zhang and
                  Jiahuan Cheng and
                  Shouhong Ding and
                  Zhengyang Zhou and
                  Yang Wang},
  editor       = {Michael J. Wooldridge and
                  Jennifer G. Dy and
                  Sriraam Natarajan},
  title        = {HDMixer: Hierarchical Dependency with Extendable Patch for Multivariate
                  Time Series Forecasting},
  booktitle    = {Thirty-Eighth {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2024, Thirty-Sixth Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2024, Fourteenth Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2014, February 20-27, 2024, Vancouver,
                  Canada},
  pages        = {12608--12616},
  publisher    = {{AAAI} Press},
  year         = {2024},
  url          = {https://doi.org/10.1609/aaai.v38i11.29155},
  doi          = {10.1609/AAAI.V38I11.29155},
  timestamp    = {Wed, 18 Mar 2026 17:07:12 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/HuangSZCDZW24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

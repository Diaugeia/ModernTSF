---
name: "WaveNet"
summary: "WaveNet is a modified integration of DeepMind's stacked dilated causal convolution architecture for the standard univariate and multivariate time-series forecasting setting. The core network applies multiple blocks of exponentially dilated causal convolutions with gated tanh/sigmoid activations and residual plus skip connections, giving a large temporal receptive field with relatively few parameters. In TSFLab the original audio-generation head is replaced with a direct multi-step regression head (via a 1×1 convolution over the skip summaries) and RevIN instance normalization is wrapped around the network for stable long-horizon forecasting."
paper: "https://arxiv.org/abs/1609.03499"
paper_title: "WaveNet: A Generative Model for Raw Audio"
venue: "arXiv preprint"
year: 2016
code: "https://github.com/GestaltCogTeam/BasicTS"
revision: "79641b1c75246ab2d8c53bb52f2ac72588be0cdc"
license: "Apache-2.0"
---
# WaveNet

WaveNet is a modified integration of DeepMind's stacked dilated causal convolution architecture for the standard univariate and multivariate time-series forecasting setting. The core network applies multiple blocks of exponentially dilated causal convolutions with gated tanh/sigmoid activations and residual plus skip connections, giving a large temporal receptive field with relatively few parameters. In TSFLab the original audio-generation head is replaced with a direct multi-step regression head (via a 1×1 convolution over the skip summaries) and RevIN instance normalization is wrapped around the network for stable long-horizon forecasting.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/1609.03499); title: WaveNet: A Generative Model for Raw Audio; venue/year: arXiv preprint / 2016
- [codebase](https://github.com/GestaltCogTeam/BasicTS); revision: `79641b1c75246ab2d8c53bb52f2ac72588be0cdc`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/WaveNet.toml`](../../../../configs/models/WaveNet.toml).

## Differences

**Clean-room implementation: confirmed.** Gated dilated causal layers and
residual/skip paths map the paper architecture; BasicTS is reference-only and
was not copied. Direct regression and RevIN are explicit forecasting changes,
and no audio-likelihood or metric reference comparison is claimed.

## Shared components

- [`gated_dilated_conv`](../_components/gated_dilated_conv/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `residual_channels=16`, `dilation_channels=16`, `skip_channels=64`, `end_channels=128`, `kernel_size=2`, `blocks=2`, `layers=2`, `use_norm=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: WaveNet: A Generative Model for Raw Audio
- **Venue**: arXiv preprint
- **Published**: 2016
- **arXiv**: https://arxiv.org/abs/1609.03499

## Abstract
This paper introduces WaveNet, a deep neural network for generating raw audio waveforms. The model is fully probabilistic and autoregressive, with the predictive distribution for each audio sample conditioned on all previous ones; nonetheless we show that it can be efficiently trained on data with tens of thousands of samples per second of audio. When applied to text-to-speech, it yields state-of-the-art performance, with human listeners rating it as significantly more natural sounding than the best parametric and concatenative systems for both English and Mandarin. A single WaveNet can capture the characteristics of many different speakers with equal fidelity, and can switch between them by conditioning on the speaker identity. When trained to model music, we find that it generates novel and often highly realistic musical fragments. We also show that it can be employed as a discriminative model, returning promising results for phoneme recognition.

## In TSFLab
Default config: `configs/models/WaveNet.toml`; model specification: `spec.py`;
clean-room implementation: `model.py`.

## Verification

**Clean-room implementation: confirmed.** Gated dilated causal layers and
residual/skip paths map the paper architecture; BasicTS is reference-only and
was not copied. Direct regression and RevIN are explicit forecasting changes,
and no audio-likelihood or metric reference comparison is claimed.

## Citation

```bibtex
@inproceedings{DBLP:conf/ssw/OordDZSVGKSK16,
  author       = {A{\"{a}}ron van den Oord and
                  Sander Dieleman and
                  Heiga Zen and
                  Karen Simonyan and
                  Oriol Vinyals and
                  Alex Graves and
                  Nal Kalchbrenner and
                  Andrew W. Senior and
                  Koray Kavukcuoglu},
  editor       = {Alan W. Black},
  title        = {WaveNet: {A} Generative Model for Raw Audio},
  booktitle    = {The 9th {ISCA} Speech Synthesis Workshop, {SSW} 2016, Sunnyvale, CA,
                  USA, September 13-15, 2016},
  pages        = {125},
  publisher    = {{ISCA}},
  year         = {2016},
  url          = {https://www.isca-archive.org/ssw\_2016/vandenoord16\_ssw.html},
  timestamp    = {Wed, 31 Jul 2024 16:45:19 +0200},
  biburl       = {https://dblp.org/rec/conf/ssw/OordDZSVGKSK16.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

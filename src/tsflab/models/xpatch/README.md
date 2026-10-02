---
name: "xPatch"
summary: "xPatch is a dual-stream forecaster that applies an exponential (EMA or Holt double smoothing) seasonal-trend decomposition and feeds the trend to a linear MLP stream and the seasonal part to a patch-based depthwise-CNN stream, both channel-independent, then fuses the two forecasts with an MLP. The paper's arctangent loss and sigmoid learning-rate schedule are not implemented; the default loss is MSE."
paper: "https://arxiv.org/abs/2412.17323"
paper_title: "xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition"
venue: "AAAI 2025"
year: 2025
code: "https://github.com/stitsyuk/xPatch"
revision: "d12eecaa11409109582f5e2ffdebcc2cffd47b3e"
license: "Apache-2.0"
tagline: "Exponential smoothing seasonal-trend split feeding a linear trend MLP stream and a patch-CNN seasonal stream, fused."
tags: ["mlp", "cnn", "decomposition", "patching", "channel-independent", "normalization"]
composition: ["normalization=component:revin", "decomposition=local:exponential-moving-average-decomposition", "temporal=local:linear-trend-stream+local:patch-cnn-seasonal-stream", "channel=local:channel-independent-shared-weights", "head=local:fusion-mlp", "loss=loss:mse"]
---
# xPatch

## Key ideas

- `ExponentialDecomposition` (EMA, or Holt double smoothing with `ma_type=dema`) splits the normalized series into seasonal and trend parts with a recurrence rather than a moving-average kernel.
- `LinearTrendStream` is a pooled, LayerNorm'd MLP on the trend; `NonlinearPatchStream` embeds seasonal patches then applies depthwise and pointwise Conv1d with batch norm and a residual.
- `DualStreamForecaster` concatenates both stream forecasts and fuses them with an MLP; channels are folded into the batch and `revin` wraps the model.
- The paper's arctangent loss and sigmoid learning-rate schedule are not part of the model code; the default loss is MSE.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2412.17323); title: xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition; venue/year: AAAI 2025 / 2025
- [codebase](https://github.com/stitsyuk/xPatch); revision: `d12eecaa11409109582f5e2ffdebcc2cffd47b3e`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/xPatch.toml`](../../../../configs/models/xPatch.toml).

## Differences

Implementation: **rewrite**. The former official reference reference comparison attempt was blocked by
the pinned implementation's CUDA-only EMA path. The current implementation is
an independent, device-neutral reconstruction from the paper; the linked code
repository is reference-only and its implementation source was not copied.
Clean-room implementation: confirmed.

The rewrite implements paper equation (2), `s[0]=x[0]` and
`s[t]=alpha*x[t]+(1-alpha)*s[t-1]`, followed by `seasonal=x-trend`. The trend
uses an activation-free two-stage linear/pooling/normalization bottleneck. The
seasonal branch unfolds channel-independent patches and applies an embedding,
depthwise convolution, residual pooling, pointwise convolution, and MLP head;
the two horizon representations are learnedly fused.

Material differences: layer widths are explicit local defaults because the
paper does not fully specify every hidden dimension; end padding is last-value
replication; the optional `dema` route uses the standard
Holt level-and-trend recurrence controlled by `alpha` and `beta`, and is not the
paper's default EMA experiment. The paper's arctangent loss and sigmoid
learning-rate schedule are training policies and are not embedded in this
forecasting module.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `patch_len=16`, `stride=8`, `padding_patch='end'`, `ma_type='ema'`, `alpha=0.3`, `beta=0.3`, `revin=True`, `hidden_dim=64`
<!-- model-card:canonical:end -->

## Paper
- **Title**: xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition
- **Venue**: AAAI 2025
- **Published**: 2025 (arXiv: 2024-12)
- **arXiv**: https://arxiv.org/abs/2412.17323

## Abstract
In recent years, the application of transformer-based models in time-series forecasting has received significant attention. While often demonstrating promising results, the transformer architecture encounters challenges in fully exploiting the temporal relations within time series data due to its attention mechanism. In this work, we design eXponential Patch (xPatch for short), a novel dual-stream architecture that utilizes exponential decomposition. Inspired by the classical exponential smoothing approaches, xPatch introduces the innovative seasonal-trend exponential decomposition module. Additionally, we propose a dual-flow architecture that consists of an MLP-based linear stream and a CNN-based non-linear stream. This model investigates the benefits of employing patching and channel-independence techniques within a non-transformer model. Finally, we develop a robust arctangent loss function and a sigmoid learning rate adjustment scheme, which prevent overfitting and boost forecasting performance.

## In TSFLab
Default config: `configs/models/xPatch.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

Implementation: **rewrite**. The former official reference reference comparison attempt was blocked by
the pinned implementation's CUDA-only EMA path. The current implementation is
an independent, device-neutral reconstruction from the paper; the linked code
repository is reference-only and its implementation source was not copied.
Clean-room implementation: confirmed.

The rewrite implements paper equation (2), `s[0]=x[0]` and
`s[t]=alpha*x[t]+(1-alpha)*s[t-1]`, followed by `seasonal=x-trend`. The trend
uses an activation-free two-stage linear/pooling/normalization bottleneck. The
seasonal branch unfolds channel-independent patches and applies an embedding,
depthwise convolution, residual pooling, pointwise convolution, and MLP head;
the two horizon representations are learnedly fused.

Material differences: layer widths are explicit local defaults because the
paper does not fully specify every hidden dimension; end padding is last-value
replication; the optional `dema` route uses the standard
Holt level-and-trend recurrence controlled by `alpha` and `beta`, and is not the
paper's default EMA experiment. The paper's arctangent loss and sigmoid
learning-rate schedule are training policies and are not embedded in this
forecasting module.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/StitsyukC25,
  author       = {Artyom Stitsyuk and
                  Jaesik Choi},
  editor       = {Toby Walsh and
                  Julie Shah and
                  Zico Kolter},
  title        = {xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend
                  Decomposition},
  booktitle    = {Thirty-Ninth {AAAI} Conference on Artificial Intelligence, Thirty-Seventh
                  Conference on Innovative Applications of Artificial Intelligence,
                  Fifteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2025, Philadelphia, PA, USA, February 25 - March 4, 2025},
  pages        = {20601--20609},
  publisher    = {{AAAI} Press},
  year         = {2025},
  url          = {https://doi.org/10.1609/aaai.v39i19.34270},
  doi          = {10.1609/AAAI.V39I19.34270},
  timestamp    = {Wed, 18 Mar 2026 17:07:12 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/StitsyukC25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

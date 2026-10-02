---
name: "FreDF"
summary: "FreDF zero-pads the history by the horizon and, per block, applies a separate complex linear transfer to every Fourier bin before fusing the bins with learned weights."
paper: "https://arxiv.org/abs/2407.12415"
paper_title: "Not All Frequencies Are Created Equal: Towards a Dynamic Fusion of Frequencies in Time-Series Forecasting"
venue: "ACM MM 2024"
year: 2024
code: "https://github.com/Zh-XY22/FreDF"
revision: "43ba9576f8ef7ccc75e046c8deca08baa7eb0384"
license: "unlicensed (no LICENSE file in repository; inspected only for read-only paper-structure clarification, no source copied)"
tagline: "Zero-padded series, per-frequency complex transfer matrices, learned frequency-fusion weights, residual FDBlocks."
tags: ["mlp", "frequency", "fourier", "frequency-fusion", "normalization", "channel-mixing", "time-series", "revin", "embed", "marks"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=local:per-frequency-transfer-fusion", "channel=local:feature-embedding-mixing", "head=local:horizon-mlp-projection", "loss=none"]
---
# FreDF

## Key ideas

- The forecast is a learned transfer function per Fourier component: history is zero-padded by the horizon and `FrequencyDynamicFusionBlock` multiplies each rFFT bin by its own complex `d_model x d_model` matrix (paper eq. 6).
- The per-bin outputs are returned to the time domain and summed with a trainable weight vector `frequency_weight` (eq. 14); the code evaluates the K masked copies of Algorithm 1 as one weighted spectrum, with `decoupled_reference` kept as the literal form.
- Blocks are residual (`x + dropout(block(x))`); the forecast is the last `pred_len` steps, passed through a horizon MLP and a `d_model -> channels` projection.
- Inputs are instance-normalized with `revin` (no affine) and embedded with the Time-Series-Library `DataEmbedding` using optional calendar marks.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2407.12415); title: Not All Frequencies Are Created Equal: Towards a Dynamic Fusion of Frequencies in Time-Series Forecasting; venue/year: ACM MM 2024 / 2024
- [codebase](https://github.com/Zh-XY22/FreDF); revision: `43ba9576f8ef7ccc75e046c8deca08baa7eb0384`; license: `unlicensed (no LICENSE file in repository; inspected only for read-only paper-structure clarification, no source copied)`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/FreDF.toml`](../../../../configs/models/FreDF.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

- [`embed`](../_components/embed/README.md)
- [`marks`](../_components/marks/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `d_model=128`, `e_layers=1`, `dropout=0.1`, `freq='h'`
<!-- model-card:canonical:end -->

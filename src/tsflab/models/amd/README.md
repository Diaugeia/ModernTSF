---
name: "AMD"
summary: "AMD is an MLP forecaster that decomposes the history into multiple temporal scales and mixes them with a residual avg-pool pyramid (MDM), refines it with patch-wise sequential temporal aggregation and an optional channel MLP (DDI), and synthesizes the forecast from a mixture of expert predictors whose noisy top-k gate reads the multi-scale embedding (AMS), balanced by an importance loss."
paper: "https://arxiv.org/abs/2406.03751"
paper_title: "Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting"
venue: "AAAI 2025"
year: 2025
code: "https://github.com/TROUBADOUR000/AMD"
revision: "000d377a1ed8946aa817ff357cdf1de64b99abb9"
license: "MIT"
tagline: "Avg-pool multi-scale mixing, patch-sequential blocks, and an MoE of MLP predictors gated by the scale embedding."
tags: ["mlp", "multi-scale", "mixture-of-experts", "channel-mixing", "normalization", "aux-loss"]
composition: ["normalization=component:revin", "decomposition=local:multi-scale-decomposable-mixing", "temporal=local:patch-sequential-dependency-interaction", "channel=local:optional-channel-mlp", "head=local:adaptive-multi-predictor-synthesis", "loss=loss:mse+local:expert-importance-balance"]
---
# AMD

## Key ideas

- `MultiScaleDecomposableMixing` average-pools the history at windows `c^k ... c`, and adds each coarse scale's MLP output to the next finer scale, yielding a multi-scale embedding of the series.
- `DualDependencyInteraction` walks the series in `patch`-wide slices: each slice is the linear aggregate of the previous output slice plus the input slice, with an optional `alpha`-scaled MLP across channels.
- `AdaptiveMultiPredictorSynthesis` mixes `num_experts` shared `seq_len -> pred_len` MLP predictors per channel; a noisy gate on the MDM embedding compresses non-top-k weights, and a coefficient-of-variation importance loss (`aux_loss`) balances experts.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2406.03751); title: Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting; venue/year: AAAI 2025 / 2025
- [codebase](https://github.com/TROUBADOUR000/AMD); revision: `000d377a1ed8946aa817ff357cdf1de64b99abb9`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/AMD.toml`](../../../../configs/models/AMD.toml).

## Differences

Paper and official code (pinned revision `000d377`, `models/tsAMD.py`, `models/common.py`, `models/tsmoe.py`) were checked; the local module is an independent rewrite. MDM, DDI, the noisy gate (log1p below the k-th logit, expm1 above, scaled by 10, then softmax) and the importance loss follow the official modules. Recorded differences:

- The official `AMS` loops over channels in Python with one shared gate and one shared expert set; the local module computes all channels in one batched pass (same math, same shared parameters). The gate input is the MDM time embedding and the experts read the DDI output, as in the official model.
- The importance loss reproduces the official `cv_squared`: gates are expanded over the horizon before the batch sum, so the variance uses the `experts * pred_len` element count, summed over channels. It is exposed as `model.aux_loss` (scaled by `moe_loss_coef`, default 1.0 like the official `loss_coef`) in training mode only and is added to the criterion by the shared trainer; it is `None` in evaluation. The official `forward` returns `(forecast, moe_loss)` and its loop adds the loss only in training.
- Official hard-codes `ff_dim=2048`, `num_experts=8`, `top_k=2` and calls the BatchNorm switch `layernorm`; TSFLab exposes them as `ff_dim`, `num_experts`, `top_k` and `batchnorm` (the normalization is BatchNorm1d over flattened `channels * length` features, so `enc_in` and `seq_len` are fixed at construction). The official `target_slice` output selection is dropped.
- Shape validation is added: `seq_len` must be divisible by `patch` and by `mix_layer_scale**mix_layer_num`, and `top_k <= num_experts`. The official scripts use `seq_len=512` while the contract fixture uses 96; the preset values (`n_block=1`, `alpha=0.0`, `mix_layer_num=3`, `patch=16`, `dropout=0.1`) follow the official scripts.
- RevIN is the shared `revin` component with affine parameters, matching the official RevIN.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `n_block=1`, `alpha=0.0`, `mix_layer_num=3`, `mix_layer_scale=2`, `patch=16`, `dropout=0.1`, `num_experts=8`, `top_k=2`, `ff_dim=2048`, `moe_loss_coef=1.0`, `use_revin=True`, `batchnorm=True`
<!-- model-card:canonical:end -->

## Source and verification

Paper and official code (pinned revision `000d377`, `models/tsAMD.py`, `models/common.py`, `models/tsmoe.py`) were checked; the local module is an independent rewrite. MDM, DDI, the noisy gate (log1p below the k-th logit, expm1 above, scaled by 10, then softmax) and the importance loss follow the official modules. Recorded differences:

- The official `AMS` loops over channels in Python with one shared gate and one shared expert set; the local module computes all channels in one batched pass (same math, same shared parameters). The gate input is the MDM time embedding and the experts read the DDI output, as in the official model.
- The importance loss reproduces the official `cv_squared`: gates are expanded over the horizon before the batch sum, so the variance uses the `experts * pred_len` element count, summed over channels. It is exposed as `model.aux_loss` (scaled by `moe_loss_coef`, default 1.0 like the official `loss_coef`) in training mode only and is added to the criterion by the shared trainer; it is `None` in evaluation. The official `forward` returns `(forecast, moe_loss)` and its loop adds the loss only in training.
- Official hard-codes `ff_dim=2048`, `num_experts=8`, `top_k=2` and calls the BatchNorm switch `layernorm`; TSFLab exposes them as `ff_dim`, `num_experts`, `top_k` and `batchnorm` (the normalization is BatchNorm1d over flattened `channels * length` features, so `enc_in` and `seq_len` are fixed at construction). The official `target_slice` output selection is dropped.
- Shape validation is added: `seq_len` must be divisible by `patch` and by `mix_layer_scale**mix_layer_num`, and `top_k <= num_experts`. The official scripts use `seq_len=512` while the contract fixture uses 96; the preset values (`n_block=1`, `alpha=0.0`, `mix_layer_num=3`, `patch=16`, `dropout=0.1`) follow the official scripts.
- RevIN is the shared `revin` component with affine parameters, matching the official RevIN.

## Citation

```bibtex
@inproceedings{hu2025adaptive,
  title     = {Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting},
  author    = {Hu, Yifan and Liu, Peiyuan and Zhu, Peng and Cheng, Dawei and Dai, Tao},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025}
}
```

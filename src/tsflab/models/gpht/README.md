---
name: "GPHT"
summary: "GPHT models tokens auto-regressively with a cascade of max-pooled causal patch transformers whose stage inputs are iterative residuals and whose outputs are summed."
paper: "https://arxiv.org/abs/2402.16516"
paper_title: "Generative Pretrained Hierarchical Transformer for Time Series Forecasting"
venue: "KDD 2024"
year: 2024
code: "https://github.com/icantnamemyself/GPHT"
revision: "cfdbbd2f8386603476d0b48705b6189cfae0ad90"
license: "MIT"
tagline: "Auto-regressive token forecaster: pooled causal patch-transformer stages chained by residuals, outputs summed."
tags: ["transformer", "patching", "hierarchical", "multi-scale", "autoregressive", "generative", "pretraining", "normalization", "channel-independent", "time-series", "revin", "embed", "self_attention_family", "transformer_encdec"]
composition: ["normalization=component:revin", "decomposition=local:iterative-residual-stages", "temporal=component:self_attention_family+component:transformer_encdec", "channel=local:channel-independent-shared-weights", "head=local:next-token-linear-head-rollout", "loss=local:next-token-teacher-forced-mse"]
---
# GPHT

## Key ideas

- GPHT is auto-regressive over tokens of `token_len` steps: a causally masked patch transformer maps every token to the next token (`HierarchicalStage`, paper eq. 3).
- Each stage max-pools its input (`pooling_rates`, default 8/4/2/1) so early stages see coarse patterns; stage inputs are iterative residuals `x - PadFirstToken(out)` and stage outputs are summed (eqs. 4-5).
- `revin` instance normalization wraps the stack; `forward` rolls the window one token at a time for horizons longer than `token_len`.
- `training_objective` is the paper's teacher-forced next-token MSE over every position; the pretrained mixed-dataset checkpoint is not part of this entry.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 48, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2402.16516); title: Generative Pretrained Hierarchical Transformer for Time Series Forecasting; venue/year: KDD 2024 / 2024
- [codebase](https://github.com/icantnamemyself/GPHT); revision: `cfdbbd2f8386603476d0b48705b6189cfae0ad90`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/GPHT.toml`](../../../../configs/models/GPHT.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

- [`embed`](../_components/embed/README.md)
- [`revin`](../_components/revin/README.md)
- [`self_attention_family`](../_components/self_attention_family/README.md)
- [`transformer_encdec`](../_components/transformer_encdec/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=48`. Default
model parameters are: `enc_in=7`, `token_len=48`, `pooling_rates=[8, 4, 2, 1]`, `d_model=512`, `d_ff=2048`, `n_heads=8`, `e_layers=3`, `dropout=0.1`, `activation='gelu'`
<!-- model-card:canonical:end -->

---
name: "PENGUIN"
summary: "PENGUIN is a channel-independent patch Transformer whose self-attention is a periodic-nested group attention: query heads are split into groups that each share one key/value head and add an ALiBi-style bias built from a triangle wave of the patch distance modulo that group's period, so one head group per known cycle (for example daily and weekly) sees periodic structure while a causal mask and a flatten-linear head produce the forecast."
paper: "https://arxiv.org/abs/2508.13773"
paper_title: "PENGUIN: Enhancing Transformer with Periodic-Nested Group Attention for Long-term Time Series Forecasting"
venue: "AISTATS 2026"
year: 2026
code: "https://github.com/ysygMhdxw/AISTATS2026_PENGUIN"
revision: "d9a2b84f9fa214c91c4c1c51e5a7f95fb451cefb"
license: "NOASSERTION"
tagline: "Grouped multi-query patch attention with a per-group periodic ALiBi bias, one head group per cycle length."
tags: ["transformer", "patching", "periodicity", "attention-variant", "channel-independent", "normalization", "revin"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=component:periodic_alibi_bias+local:grouped-multi-query-causal-attention-encoder", "channel=local:channel-independent-shared-weights", "head=component:flatten_forecast_head", "loss=loss:mse"]
---
# PENGUIN

## Key ideas

- `PeriodicNestedGroupAttention` splits `n_heads` into one group per period; each group shares a single key/value head (grouped multi-query attention) and keeps its own relative-position bias.
- `periodic_alibi_bias` gives head `k` of a group the bias `-m_k * tri(|i - j| mod P_S)` with `m_k = 2^(-8k/n)`; periods are given in time steps and converted to patch units by dividing by `stride`.
- An encoder layer is attention then residual and norm, ReLU/GELU feed-forward then residual and norm, with causal masking over patch tokens and a final BatchNorm; `use_rmsnorm` swaps LayerNorm for RMSNorm.
- Inputs are RevIN-normalized (`revin`), embedded per channel by replicate-padded patching (`embed.PatchEmbedding`), and read out by `flatten_forecast_head`.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2508.13773); title: PENGUIN: Enhancing Transformer with Periodic-Nested Group Attention for Long-term Time Series Forecasting; venue/year: AISTATS 2026 / 2026
- [codebase](https://github.com/ysygMhdxw/AISTATS2026_PENGUIN); revision: `d9a2b84f9fa214c91c4c1c51e5a7f95fb451cefb`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/PENGUIN.toml`](../../../../configs/models/PENGUIN.toml).

## Differences

Paper and official code (pinned revision `d9a2b84`, `models/PENGIUN.py`, `layers/SelfAttention_Family.py`, `layers/PENGIUN_EncDec.py`) were consulted for structure only; the official repository has no license file and nothing was copied. Recorded differences:

- Period units: the official model passes `periods` straight into the bias as token counts (its scripts use `patch_len=1, stride=1`). The local model takes `periods` in time steps and converts to patch units with `P_S = P / stride` (paper Sec. 4.2.2), requiring every period to be a positive multiple of `stride`.
- Bias options: the official `--alibi` and `--alibicycle` flags are always on here (`alibi=True` adds the bias; periodic distances are used whenever `periods` is non-empty). The official slope construction for head counts that are not a power of two (extra interleaved slopes) is not reproduced; the local slopes are `2^(-8k/n)` per group of `n` heads, which equals the official slopes whenever `n_heads` is a power of two (the default `n_heads=8` with one period).
- `causal` is an added switch (default True, the official always masks); the official `use_rcf` recurrent-cycle option (CycleNet-style cycle removal needing a `cycle_index` input) and the `visual` attention-weight output are not implemented.
- Normalization is the shared `revin` component without affine parameters, equal to the official mean/std normalization (`+1e-5`). The feed-forward sub-layer uses Linear layers in place of the official kernel-1 Conv1d (same math); `use_rmsnorm` uses the shared RMSNorm (`eps=1e-6`) in place of `LlamaRMSNorm` (computed in the input dtype rather than float32).
- Defaults (`d_model=128`, `d_ff=256`, `e_layers=2`, `patch_len=16`, `stride=8`, `periods=[24]`) differ from the official ETTh1 script (`d_model=16`, `d_ff=128`, `e_layers=3`, `patch_len=1`, `stride=1`, `seq_len=336`, `factor=3`); `n_heads` must be a multiple of `len(periods)`.

## Shared components

- [`embed`](../_components/embed/README.md)
- [`flatten_forecast_head`](../_components/flatten_forecast_head/README.md)
- [`mamba`](../_components/mamba/README.md)
- [`periodic_alibi_bias`](../_components/periodic_alibi_bias/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `d_model=128`, `d_ff=256`, `n_heads=8`, `e_layers=2`, `patch_len=16`, `stride=8`, `dropout=0.1`, `periods=[24]`, `activation='relu'`, `use_rmsnorm=False`, `alibi=True`, `causal=True`
<!-- model-card:canonical:end -->

## Source and verification

Paper and official code (pinned revision `d9a2b84`, `models/PENGIUN.py`, `layers/SelfAttention_Family.py`, `layers/PENGIUN_EncDec.py`) were consulted for structure only; the official repository has no license file and nothing was copied. Recorded differences:

- Period units: the official model passes `periods` straight into the bias as token counts (its scripts use `patch_len=1, stride=1`). The local model takes `periods` in time steps and converts to patch units with `P_S = P / stride` (paper Sec. 4.2.2), requiring every period to be a positive multiple of `stride`.
- Bias options: the official `--alibi` and `--alibicycle` flags are always on here (`alibi=True` adds the bias; periodic distances are used whenever `periods` is non-empty). The official slope construction for head counts that are not a power of two (extra interleaved slopes) is not reproduced; the local slopes are `2^(-8k/n)` per group of `n` heads, which equals the official slopes whenever `n_heads` is a power of two (the default `n_heads=8` with one period).
- `causal` is an added switch (default True, the official always masks); the official `use_rcf` recurrent-cycle option (CycleNet-style cycle removal needing a `cycle_index` input) and the `visual` attention-weight output are not implemented.
- Normalization is the shared `revin` component without affine parameters, equal to the official mean/std normalization (`+1e-5`). The feed-forward sub-layer uses Linear layers in place of the official kernel-1 Conv1d (same math); `use_rmsnorm` uses the shared RMSNorm (`eps=1e-6`) in place of `LlamaRMSNorm` (computed in the input dtype rather than float32).
- Defaults (`d_model=128`, `d_ff=256`, `e_layers=2`, `patch_len=16`, `stride=8`, `periods=[24]`) differ from the official ETTh1 script (`d_model=16`, `d_ff=128`, `e_layers=3`, `patch_len=1`, `stride=1`, `seq_len=336`, `factor=3`); `n_heads` must be a multiple of `len(periods)`.

## Citation

```bibtex
@article{sun2025penguin,
  title   = {{PENGUIN}: Enhancing Transformer with Periodic-Nested Group Attention for Long-term Time Series Forecasting},
  author  = {Sun, Tian and Chen, Yuqi and Sun, Weiwei},
  journal = {arXiv preprint arXiv:2508.13773},
  year    = {2025}
}
```

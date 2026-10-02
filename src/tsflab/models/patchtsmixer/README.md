---
name: "PatchTSMixer"
summary: "PatchTSMixer (the KDD 2023 TSMixer of IBM Research) is a lightweight MLP-Mixer forecaster: RevIN, overlapping patches embedded by one linear layer, stacked layers that mix across patches, hidden features, and optionally channels with MLPs, layer norm, residuals and softmax gated attention, then a flatten-linear head shared by all channels."
paper: "https://arxiv.org/abs/2306.09364"
paper_title: "TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting"
venue: "KDD 2023"
year: 2023
code: "https://github.com/huggingface/transformers"
revision: "3693f8d26311305e914735a6373fb03468d6aaa0"
license: "Apache-2.0"
tagline: "Patch MLP-Mixer: inter-patch, feature and optional channel MLP mixing, each softmax-gated, with a shared linear head."
tags: ["mlp", "mixer", "patching", "gated-attention", "channel-independent", "normalization", "revin"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=local:patch-and-feature-mlp-mixer-layers", "channel=local:channel-independent-or-inter-channel-mixer", "head=component:flatten_forecast_head", "loss=loss:mse"]
---
# PatchTSMixer

## Key ideas

- Input is RevIN-normalized (`revin`, no affine), cut into overlapping patches by `Model.patchify`, and embedded by one shared `patch_embedding` linear layer, giving `[batch, channels, patches, d_model]`.
- Each `MixerLayer` applies `AxisMixer` blocks: pre-LayerNorm, an MLP along one axis (patches, hidden features, or, in `mix_channel` mode, channels), a `softmax_gate` gated-attention block, and a residual.
- `mode="common_channel"` (CI-TSMixer) shares all weights across channels and never mixes them; `mode="mix_channel"` (IC-TSMixer) adds the inter-channel mixer before the patch mixer in every layer.
- The prediction head dropout-regularizes, flattens patches and features, and projects with one Linear shared across channels (`flatten_forecast_head`); RevIN then restores the scale.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2306.09364); title: TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting; venue/year: KDD 2023 / 2023
- [codebase](https://github.com/huggingface/transformers); revision: `3693f8d26311305e914735a6373fb03468d6aaa0`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/PatchTSMixer.toml`](../../../../configs/models/PatchTSMixer.toml).

## Differences

Paper and the Hugging Face `transformers` implementation (pinned revision `3693f8d`, `models/patchtsmixer/modeling_patchtsmixer.py`, Apache-2.0) were checked; the local module is an independent rewrite of the supervised forecasting path. Patching (latest samples kept, `(seq_len - patch_len) // stride + 1` patches), the patch embedding, layer order (channel mixer in `mix_channel` mode, then patch mixer, then feature mixer), pre-LayerNorm residual blocks and the flatten-linear head follow the Hugging Face model. Recorded differences:

- Channel mixer gate order: Hugging Face `PatchTSMixerChannelFeatureMixerBlock` applies the gated attention before the MLP, whereas its patch and feature mixers apply it after. The local `AxisMixer` applies the gate after the MLP for all three axes (the local reading of paper Sec. 3.3.5), so `mix_channel` differs from Hugging Face in this one place.
- Normalization is the shared `revin` component without affine parameters, which matches the Hugging Face default `scaling="std"` (mean and biased standard deviation with `+1e-5`) applied per instance and channel; other scalers (`mean`, none), `observed_mask` handling and `norm_mlp="BatchNorm"` are not implemented (LayerNorm over `d_model`, `eps=1e-5`).
- Not implemented: the optional patch self-attention (`self_attn`), positional encoding, self-supervised masked pretraining, the `flatten` mode, the linear/classification/regression heads, distribution (Student-t) output, `prediction_channel_indices`, and `output_range`. Only MSE point forecasting is supported.
- Defaults differ from Hugging Face (`d_model=16` vs 8, `dropout` and `head_dropout` 0.1 vs 0.2, `patch_len=16` vs 8, stride 8) and Hugging Face's `init_std=0.02` initialization is not applied (PyTorch default initialization). The head applies dropout before flattening, which is element-wise equivalent to Hugging Face's dropout after flattening.
- The paper's online reconciliation head and hybrid channel modeling are not included.

## Shared components

- [`flatten_forecast_head`](../_components/flatten_forecast_head/README.md)
- [`revin`](../_components/revin/README.md)
- [`softmax_gate`](../_components/softmax_gate/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `d_model=16`, `patch_len=16`, `stride=8`, `num_layers=3`, `expansion_factor=2`, `dropout=0.1`, `head_dropout=0.1`, `mode='common_channel'`, `gated_attn=True`
<!-- model-card:canonical:end -->

## Source and verification

Paper and the Hugging Face `transformers` implementation (pinned revision `3693f8d`, `models/patchtsmixer/modeling_patchtsmixer.py`, Apache-2.0) were checked; the local module is an independent rewrite of the supervised forecasting path. Patching (latest samples kept, `(seq_len - patch_len) // stride + 1` patches), the patch embedding, layer order (channel mixer in `mix_channel` mode, then patch mixer, then feature mixer), pre-LayerNorm residual blocks and the flatten-linear head follow the Hugging Face model. Recorded differences:

- Channel mixer gate order: Hugging Face `PatchTSMixerChannelFeatureMixerBlock` applies the gated attention before the MLP, whereas its patch and feature mixers apply it after. The local `AxisMixer` applies the gate after the MLP for all three axes (the local reading of paper Sec. 3.3.5), so `mix_channel` differs from Hugging Face in this one place.
- Normalization is the shared `revin` component without affine parameters, which matches the Hugging Face default `scaling="std"` (mean and biased standard deviation with `+1e-5`) applied per instance and channel; other scalers (`mean`, none), `observed_mask` handling and `norm_mlp="BatchNorm"` are not implemented (LayerNorm over `d_model`, `eps=1e-5`).
- Not implemented: the optional patch self-attention (`self_attn`), positional encoding, self-supervised masked pretraining, the `flatten` mode, the linear/classification/regression heads, distribution (Student-t) output, `prediction_channel_indices`, and `output_range`. Only MSE point forecasting is supported.
- Defaults differ from Hugging Face (`d_model=16` vs 8, `dropout` and `head_dropout` 0.1 vs 0.2, `patch_len=16` vs 8, stride 8) and Hugging Face's `init_std=0.02` initialization is not applied (PyTorch default initialization). The head applies dropout before flattening, which is element-wise equivalent to Hugging Face's dropout after flattening.
- The paper's online reconciliation head and hybrid channel modeling are not included.

## Citation

```bibtex
@inproceedings{ekambaram2023tsmixer,
  title     = {{TSMixer}: Lightweight {MLP}-Mixer Model for Multivariate Time Series Forecasting},
  author    = {Ekambaram, Vijay and Jati, Arindam and Nguyen, Nam and Sinthong, Phanwadee and Kalagnanam, Jayant},
  booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2023}
}
```

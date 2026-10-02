---
name: "SAMformer"
summary: "SAMformer is a deliberately shallow Transformer for multivariate forecasting: RevIN, one single-head channel-wise attention block with a residual connection, and a linear forecaster, trained with sharpness-aware minimization (SAM) to escape the sharp loss minima that make ordinary Transformers generalize poorly on long-horizon benchmarks."
paper: "https://arxiv.org/abs/2402.10198"
paper_title: "SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention"
venue: "ICML 2024"
year: 2024
code: "https://github.com/romilbert/samformer"
revision: "71f10eaa696f2a098798779ee14b6ecd6b69bcd9"
license: "Apache-2.0"
tagline: "One channel-wise attention block plus linear head under RevIN, trained with sharpness-aware minimization."
tags: ["transformer", "channel-mixing", "attention-variant", "normalization", "lightweight", "sam", "training-objective"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=component:channel_wise_linear", "channel=local:channel-wise-attention", "head=component:channel_wise_linear", "loss=loss:mse+component:sharpness_aware"]
---
# SAMformer

## Key ideas

- `ChannelWiseAttention` treats each variate as a token whose features are its `seq_len` time steps: `softmax(QK^T / sqrt(hid_dim)) V` with `Q, K` projected to `hid_dim` and `V` kept at width `seq_len`, added residually; a single head, a single layer.
- The attended window is mapped to the horizon by one shared linear layer (`channel_wise_linear`), and `revin` normalizes the input and restores scale on the forecast.
- Sharpness-aware minimization is the training procedure, not architecture: `spec.training_objective` evaluates the loss at `w + rho * g / ||g||` through the `sharpness_aware` component, so ordinary optimizers realize SAM; `forward` is the plain network.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2402.10198); title: SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention; venue/year: ICML 2024 / 2024
- [codebase](https://github.com/romilbert/samformer); revision: `71f10eaa696f2a098798779ee14b6ecd6b69bcd9`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/SAMformer.toml`](../../../../configs/models/SAMformer.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

- [`channel_wise_linear`](../_components/channel_wise_linear/README.md)
- [`revin`](../_components/revin/README.md)
- [`sharpness_aware`](../_components/sharpness_aware/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `hid_dim=16`, `rho=0.5`, `use_revin=True`
<!-- model-card:canonical:end -->

## Citation

```bibtex
@inproceedings{ilbert2024samformer,
  title     = {{SAMformer}: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention},
  author    = {Romain Ilbert and Ambroise Odonnat and Vasilii Feofanov and Aladin Virmaux and Giuseppe Paolo and Themis Palpanas and Ievgen Redko},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year      = {2024}
}
```

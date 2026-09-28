---
name: "BiMamba"
summary: "BiMamba is a bidirectional state-space model (SSM) for long-term multivariate time-series forecasting. It extends the Mamba selective SSM with a forget gate (Mamba+) and runs it in both the forward and backward directions, enabling the model to capture long-range temporal dependencies without the quadratic cost of Transformer attention. A series-relation-aware decider automatically selects between channel-independent and channel-mixing tokenisation strategies depending on the dataset."
paper: "https://arxiv.org/abs/2404.15772"
paper_title: "Bi-Mamba+: Bidirectional Mamba for Time Series Forecasting"
venue: "arXiv preprint"
year: 2024
code: "https://github.com/Huangmr0719/BiMamba"
revision: "78db48cc5251235e47465c63d3701a9e5fd6fcb1"
license: "NOASSERTION"
---
# BiMamba

BiMamba is a bidirectional state-space model (SSM) for long-term multivariate time-series forecasting. It extends the Mamba selective SSM with a forget gate (Mamba+) and runs it in both the forward and backward directions, enabling the model to capture long-range temporal dependencies without the quadratic cost of Transformer attention. A series-relation-aware decider automatically selects between channel-independent and channel-mixing tokenisation strategies depending on the dataset.

<!-- model-card:canonical:start -->
## Method overview

BiMamba is a bidirectional state-space model (SSM) for long-term multivariate time-series forecasting.

## Core architecture

It extends the Mamba selective SSM with a forget gate (Mamba+) and runs it in both the forward and backward directions, enabling the model to capture long-range temporal dependencies without the quadratic cost of Transformer attention. A series-relation-aware decider automatically selects between channel-independent and channel-mixing tokenisation strategies depending on the dataset.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2404.15772); title: Bi-Mamba+: Bidirectional Mamba for Time Series Forecasting; venue/year: arXiv preprint / 2024
- [codebase](https://github.com/Huangmr0719/BiMamba); revision: `78db48cc5251235e47465c63d3701a9e5fd6fcb1`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/BiMamba.toml`](../../../../configs/models/BiMamba.toml).

## Differences

**Clean-room implementation: confirmed.** The paper's patch, SRA, Mamba+ gate, bidirectional encoder, and flatten-head equations have executable structure tests. Inputs are `[B, seq_len, enc_in]` and outputs are `[B, pred_len, enc_in]`; marks are ignored. The author repository remains reference-only and no code was copied.

## Shared components

- [`mamba`](../_components/mamba/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `d_state=16`, `e_layers=2`, `expand=2`, `d_conv=4`, `dropout=0.1`, `patch_len=16`, `stride=8`, `sra_threshold=0.5`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Bi-Mamba+: Bidirectional Mamba for Time Series Forecasting
- **Venue**: arXiv preprint
- **Published**: 2024 (arXiv: 2024-04)
- **arXiv**: https://arxiv.org/abs/2404.15772

## Abstract
Long-term time series forecasting (LTSF) provides longer insights into future trends and patterns. Over the past few years, deep learning models especially Transformers have achieved advanced performance in LTSF tasks. However, LTSF faces inherent challenges such as long-term dependencies capturing and sparse semantic characteristics. Recently, a new state space model (SSM) named Mamba is proposed. With the selective capability on input data and the hardware-aware parallel computing algorithm, Mamba has shown great potential in balancing predicting performance and computational efficiency compared to Transformers. To enhance Mamba's ability to preserve historical information in a longer range, we design a novel Mamba+ block by adding a forget gate inside Mamba to selectively combine the new features with the historical features in a complementary manner. Furthermore, we apply Mamba+ both forward and backward and propose Bi-Mamba+, aiming to promote the model's ability to capture interactions among time series elements. Additionally, multivariate time series data in different scenarios may exhibit varying emphasis on intra- or inter-series dependencies. Therefore, we propose a series-relation-aware decider that controls the utilization of channel-independent or channel-mixing tokenization strategy for specific datasets. Extensive experiments on 8 real-world datasets show that our model achieves better predictions compared with state-of-the-art methods. Our code is available at https://github.com/Leopold2333/Bi-Mamba+.

## In ModernTSF
Default config: `configs/models/BiMamba.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

**Clean-room implementation: confirmed.** The paper's patch, SRA, Mamba+ gate, bidirectional encoder, and flatten-head equations have executable structure tests. Inputs are `[B, seq_len, enc_in]` and outputs are `[B, pred_len, enc_in]`; marks are ignored. The author repository remains reference-only and no code was copied.

## Citation

```bibtex
@misc{liang2024bimamba,
  author        = {Aobo Liang and
                  Xingguo Jiang and
                  Yan Sun and
                  Xiaohou Shi and
                  Ke Li},
  title         = {Bi-Mamba+: Bidirectional Mamba for Time Series Forecasting},
  year          = {2024},
  eprint        = {2404.15772},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2404.15772}
}
```

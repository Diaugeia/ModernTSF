---
name: "FreDF"
summary: "FreDF (Frequency Dynamic Fusion, ACM MM 2024) zero-pads the history by the horizon and, per block, applies a separate complex linear transfer to every Fourier bin before fusing the bins with learned weights."
paper: "https://arxiv.org/abs/2407.12415"
paper_title: "Not All Frequencies Are Created Equal: Towards a Dynamic Fusion of Frequencies in Time-Series Forecasting"
venue: "ACM MM 2024"
year: 2024
code: "https://github.com/Zh-XY22/FreDF"
revision: "43ba9576f8ef7ccc75e046c8deca08baa7eb0384"
license: "NOASSERTION"
tagline: "Zero-padded series, per-frequency complex transfer matrices, learned frequency-fusion weights, residual FDBlocks."
tags: ["mlp", "frequency", "frequency-fusion", "normalization", "channel-mixing", "revin", "marks"]
composition: ["normalization=component:revin", "decomposition=none", "temporal=local:per-frequency-transfer-fusion", "channel=local:feature-embedding-mixing", "head=local:horizon-mlp-projection", "loss=none"]
---
# FreDF

## Key ideas

- The forecast is a learned transfer function per Fourier component: history is zero-padded by the horizon and `FrequencyDynamicFusionBlock` multiplies each rFFT bin by its own complex `d_model x d_model` matrix (paper eq. 6).
- The per-bin outputs are returned to the time domain and summed with a trainable weight vector `frequency_weight` (eq. 14); the code evaluates the K masked copies of Algorithm 1 as one weighted spectrum, with `decoupled_reference` kept as the literal form.
- Blocks are residual (`x + dropout(block(x))`); the forecast is the last `pred_len` steps, passed through a horizon MLP and a `d_model -> channels` projection.
- Paper identity: this is the architecture of arXiv 2407.12415 (Zhang et al., ACM MM 2024), not the frequency-domain loss of "FreDF: Learning to Forecast in the Frequency Domain" (arXiv 2402.02399, ICLR 2025), which is a different method that shares the name.
- Inputs are instance-normalized with `revin` (no affine) and embedded with the Time-Series-Library `DataEmbedding` using optional calendar marks.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2407.12415); title: Not All Frequencies Are Created Equal: Towards a Dynamic Fusion of Frequencies in Time-Series Forecasting; venue/year: ACM MM 2024 / 2024
- [codebase](https://github.com/Zh-XY22/FreDF); revision: `43ba9576f8ef7ccc75e046c8deca08baa7eb0384`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/FreDF.toml`](../../../../configs/models/FreDF.toml).

## Differences

- Implementation: independent local rewrite of the FreDF architecture of arXiv 2407.12415 (Frequency Dynamic Fusion, FDBlock, Algorithm 1, Eqs. 6 and 14). The official repository (`Zh-XY22/FreDF`, revision `43ba9576f8ef7ccc75e046c8deca08baa7eb0384`) has no LICENSE file, so it was read only for paper-structure clarification (`models/FreDF.py`, `run.py`) and nothing was copied; the license field records that absence rather than assuming a license.
- Resolved from the official code: the history is instance-normalized, passed through `Linear(seq_len, seq_len)` + ReLU + dropout over time, zero-padded by the horizon, embedded with the Time-Series-Library `DataEmbedding` (`timeF`) using the concatenated history and horizon marks, and each block is `x + dropout(FDBlock(x))`; the last `pred_len` steps pass through `Linear(pred_len, pred_len)` + ReLU + dropout and a `d_model -> channels` projection before de-normalization. The per-bin complex linear maps are bias-free `d_model x d_model` matrices.
- Equivalent reformulation: Algorithm 1 builds K masked copies of the spectrum, transforms each back and sums them with the frequency weights; because the inverse FFT is linear and each weight is a real scalar, `FrequencyDynamicFusionBlock` evaluates this as one weighted inverse transform. `decoupled_reference` keeps the literal K-copy form.
- Differences from the official code: (1) the official `FFT_for_Decomp` module is built once and reused by every layer, so `e_layers > 1` shares one set of transfer matrices and weights, while here each layer owns its own `FrequencyDynamicFusionBlock` (identical for the default `e_layers=1`). (2) The official fusion weights are re-softmaxed in place on every forward pass (`self.weights.data = softmax(self.weights)`), so they stay on the simplex; here `frequency_weight` is initialized from a softmax and then trained unconstrained, so entries can leave the simplex. (3) The official `irfft` uses its default length `2 * (bins - 1)`, which equals `seq_len + pred_len` only when that sum is even; here the length is passed explicitly. (4) The instance normalization is the shared `revin` (biased variance plus `1e-5`, affine off) which detaches the standard deviation, while the official code detaches only the mean. (5) Only hourly calendar marks are supported (`freq="h"`; official `--freq` accepts other values) and the embedding is always on (the official `is_embed` flag is always true in practice); when marks are absent the temporal embedding term is dropped. (6) `d_model=128`, `e_layers=1` and `dropout=0.1` match the official defaults, but its batch size 4 and learning rate 1e-4 are runner settings and are not part of the preset. Reported benchmark numbers are not reproduction claims of this implementation.
- Verification: structure and equivalence tests in `tests/test_fredf_structure.py`; no training was run.

## Shared components

- [`embed`](../_components/embed/README.md)
- [`marks`](../_components/marks/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `d_model=128`, `e_layers=1`, `dropout=0.1`, `freq='h'`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Not All Frequencies Are Created Equal: Towards a Dynamic Fusion of Frequencies in Time-Series Forecasting
- **Venue**: ACM MM 2024 (32nd ACM International Conference on Multimedia)
- **Published**: 2024 (arXiv: 2024-07)
- **arXiv**: https://arxiv.org/abs/2407.12415
- **Not this paper**: "FreDF: Learning to Forecast in the Frequency Domain" (arXiv 2402.02399, Wang et al., ICLR 2025) proposes a training loss that aligns forecasts and labels in the frequency domain to reduce label autocorrelation. It is a different method; TSFLab does not implement it here, and nothing in this entry is taken from it.

## Source and verification

- Implementation: independent local rewrite of the FreDF architecture of arXiv 2407.12415 (Frequency Dynamic Fusion, FDBlock, Algorithm 1, Eqs. 6 and 14). The official repository (`Zh-XY22/FreDF`, revision `43ba9576f8ef7ccc75e046c8deca08baa7eb0384`) has no LICENSE file, so it was read only for paper-structure clarification (`models/FreDF.py`, `run.py`) and nothing was copied; the license field records that absence rather than assuming a license.
- Resolved from the official code: the history is instance-normalized, passed through `Linear(seq_len, seq_len)` + ReLU + dropout over time, zero-padded by the horizon, embedded with the Time-Series-Library `DataEmbedding` (`timeF`) using the concatenated history and horizon marks, and each block is `x + dropout(FDBlock(x))`; the last `pred_len` steps pass through `Linear(pred_len, pred_len)` + ReLU + dropout and a `d_model -> channels` projection before de-normalization. The per-bin complex linear maps are bias-free `d_model x d_model` matrices.
- Equivalent reformulation: Algorithm 1 builds K masked copies of the spectrum, transforms each back and sums them with the frequency weights; because the inverse FFT is linear and each weight is a real scalar, `FrequencyDynamicFusionBlock` evaluates this as one weighted inverse transform. `decoupled_reference` keeps the literal K-copy form.
- Differences from the official code: (1) the official `FFT_for_Decomp` module is built once and reused by every layer, so `e_layers > 1` shares one set of transfer matrices and weights, while here each layer owns its own `FrequencyDynamicFusionBlock` (identical for the default `e_layers=1`). (2) The official fusion weights are re-softmaxed in place on every forward pass (`self.weights.data = softmax(self.weights)`), so they stay on the simplex; here `frequency_weight` is initialized from a softmax and then trained unconstrained, so entries can leave the simplex. (3) The official `irfft` uses its default length `2 * (bins - 1)`, which equals `seq_len + pred_len` only when that sum is even; here the length is passed explicitly. (4) The instance normalization is the shared `revin` (biased variance plus `1e-5`, affine off) which detaches the standard deviation, while the official code detaches only the mean. (5) Only hourly calendar marks are supported (`freq="h"`; official `--freq` accepts other values) and the embedding is always on (the official `is_embed` flag is always true in practice); when marks are absent the temporal embedding term is dropped. (6) `d_model=128`, `e_layers=1` and `dropout=0.1` match the official defaults, but its batch size 4 and learning rate 1e-4 are runner settings and are not part of the preset. Reported benchmark numbers are not reproduction claims of this implementation.
- Verification: structure and equivalence tests in `tests/test_fredf_structure.py`; no training was run.

## Citation

```bibtex
@inproceedings{zhang2024fredf,
  title         = {Not All Frequencies Are Created Equal: Towards a Dynamic Fusion of Frequencies in Time-Series Forecasting},
  author        = {Zhang, Xingyu and Zhao, Siyu and Song, Zeen and Guo, Huijie and Zhang, Jianqi and Zheng, Changwen and Qiang, Wenwen},
  booktitle     = {Proceedings of the 32nd ACM International Conference on Multimedia},
  year          = {2024},
  eprint        = {2407.12415},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2407.12415}
}
```

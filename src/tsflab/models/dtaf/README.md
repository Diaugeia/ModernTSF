---
name: "DTAF"
summary: "DTAF is a dual-branch time series forecasting framework designed to handle non-stationary data by simultaneously addressing temporal distribution shifts and spectral variability: the Temporal Stabilizing Fusion (TFS) module suppresses non-stationary temporal patterns via a mixture-of-experts filter while the Frequency Wave Modeling (FWM) module applies frequency differencing to highlight spectral shifts, with the two branches fused for robust long-term predictions."
paper: "https://arxiv.org/abs/2511.08229"
paper_title: "Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing"
venue: "AAAI 2026"
year: 2026
code: "https://github.com/decisionintelligence/DTAF"
revision: "9d12aa4061c771b419c5a5bba9f2bf95d9419c41"
license: "NOASSERTION"
tagline: "Patch Transformer branches: MoE subtraction of nuisance patterns plus top-k spectral-change frequency modeling, fused."
tags: ["transformer", "patching", "frequency", "mixture-of-experts", "channel-independent", "non-stationary"]
composition: ["normalization=local:instance-standardization", "decomposition=none", "temporal=local:temporal-stabilizing-fusion-moe+local:frequency-wave-modeling", "channel=local:channel-independent-shared-weights", "head=local:flatten-linear-head", "loss=loss:mse"]
---
# DTAF

## Key ideas

- `TemporalStabilizingFusion` routes each patch token over small expert MLPs and subtracts the weighted output as nuisance (non-stationary) content, then fuses causal history with a gated current token.
- `FrequencyWaveModeling` keeps only the DC bin and the `top_k` bins with the largest adjacent-bin amplitude change, reconstructs the series, and applies self-attention.
- `DTAFBlock` concatenates and fuses the two branches residually.
- Channels share weights through patch tokens (`patchify`); a flatten linear head forecasts, and the input is standardized per window.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2511.08229); title: Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing; venue/year: AAAI 2026 / 2026
- [codebase](https://github.com/decisionintelligence/DTAF); revision: `9d12aa4061c771b419c5a5bba9f2bf95d9419c41`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/DTAF.toml`](../../../../configs/models/DTAF.toml).

## Differences

Clean-room implementation: confirmed. The implementation was derived from the published TFS/FWM description and did not copy source from the reference-only repository. The portable expert MLP and direct forecast head are disclosed local choices.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=32`, `e_layers=1`, `patch_len=16`, `stride=8`, `heads=2`, `dropout=0.1`, `expert_num=2`, `expert_hidden=8`, `top_k=1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing
- **Venue**: AAAI 2026
- **Published**: 2026 (arXiv: 2025-11)
- **arXiv**: https://arxiv.org/abs/2511.08229

## Abstract
Time series forecasting is critical for decision-making across dynamic domains such as energy, finance, transportation, and cloud computing. However, real-world time series often exhibit non-stationarity, including temporal distribution shifts and spectral variability, which pose significant challenges for long-term time series forecasting. In this paper, we propose DTAF, a dual-branch framework that addresses non-stationarity in both the temporal and frequency domains. For the temporal domain, the Temporal Stabilizing Fusion (TFS) module employs a non-stationary mix of experts (MOE) filter to disentangle and suppress temporal non-stationary patterns while preserving long-term dependencies. For the frequency domain, the Frequency Wave Modeling (FWM) module applies frequency differencing to dynamically highlight components with significant spectral shifts. By fusing the complementary outputs of TFS and FWM, DTAF generates robust forecasts that adapt to both temporal and frequency domain non-stationarity. Extensive experiments on real-world benchmarks demonstrate that DTAF outperforms state-of-the-art baselines, yielding significant improvements in forecasting accuracy under non-stationary conditions. All codes are available at https://github.com/decisionintelligence/DTAF.

## In TSFLab
Default config: `configs/models/DTAF.toml`; model specification: `spec.py`; implementation: `model.py`.

## Source and verification

Clean-room implementation: confirmed. The implementation was derived from the published TFS/FWM description and did not copy source from the reference-only repository. The portable expert MLP and direct forecast head are disclosed local choices.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/LuCGSWY26,
  author       = {Junkai Lu and
                  Peng Chen and
                  Chenjuan Guo and
                  Yang Shu and
                  Meng Wang and
                  Bin Yang},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {Towards Non-Stationary Time Series Forecasting with Temporal Stabilization
                  and Frequency Differencing},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {24070--24078},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i29.39585},
  doi          = {10.1609/AAAI.V40I29.39585},
  timestamp    = {Wed, 25 Mar 2026 16:59:58 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/LuCGSWY26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

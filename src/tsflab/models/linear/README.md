---
name: "Linear"
summary: "Linear is one of the embarrassingly simple one-layer linear models from the LTSF-Linear family that directly maps the full historical input window to the prediction horizon via a single learnable linear projection applied independently per channel, serving as a strong baseline that outperforms complex Transformer-based models on long-term time series forecasting."
paper: "https://arxiv.org/abs/2205.13504"
paper_title: "Are Transformers Effective for Time Series Forecasting?"
venue: "AAAI 2023"
year: 2023
code: "https://github.com/cure-lab/LTSF-Linear"
revision: "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"
license: "Apache-2.0"
tagline: "Single learnable history-to-horizon linear map applied along time to each channel, optionally one map per channel."
tags: ["linear", "baseline", "lightweight", "channel-independent", "ltsf-linear"]
composition: ["normalization=none", "decomposition=none", "temporal=component:channel_wise_linear", "channel=local:channel-independent-shared-weights", "head=component:channel_wise_linear", "loss=loss:mse"]
---
# Linear

## Key ideas

- One affine map `W` of shape `pred_len x seq_len` (`ChannelWiseLinear`) forecasts every channel independently.
- Weights are shared across channels by default; `individual=True` gives each channel its own map (a repository extension).
- No normalization, decomposition or nonlinearity; the plain baseline behind DLinear and NLinear.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2205.13504); title: Are Transformers Effective for Time Series Forecasting?; venue/year: AAAI 2023 / 2023
- [codebase](https://github.com/cure-lab/LTSF-Linear); revision: `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`; license: `Apache-2.0`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/Linear.toml`](../../../../configs/models/Linear.toml).

## Differences

**Paper-driven local implementation.** The model implements the paper equation `X_hat_i = W X_i` as one temporal affine map shared across channels. The optional `individual` setting is a documented repository extension with one map per channel. The external repository is reference-only; no source file was copied or adapted. Paper preprocessing and reported benchmark values are outside code-validation scope.

## Shared components

- [`channel_wise_linear`](../_components/channel_wise_linear/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `individual=False`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Are Transformers Effective for Time Series Forecasting?
- **Venue**: AAAI 2023
- **Published**: 2023 (arXiv: 2022-05)
- **arXiv**: https://arxiv.org/abs/2205.13504

## Abstract
Recently, there has been a surge of Transformer-based solutions for the long-term time series forecasting (LTSF) task. Despite the growing performance over the past few years, we question the validity of this line of research in this work. Specifically, Transformers is arguably the most successful solution to extract the semantic correlations among the elements in a long sequence. However, in time series modeling, we are to extract the temporal relations in an ordered set of continuous points. While employing positional encoding and using tokens to embed sub-series in Transformers facilitate preserving some ordering information, the nature of the permutation-invariant self-attention mechanism inevitably results in temporal information loss. To validate our claim, we introduce a set of embarrassingly simple one-layer linear models named LTSF-Linear for comparison. Experimental results on nine real-life datasets show that LTSF-Linear surprisingly outperforms existing sophisticated Transformer-based LTSF models in all cases, and often by a large margin. Moreover, we conduct comprehensive empirical studies to explore the impacts of various design elements of LTSF models on their temporal relation extraction capability. We hope this surprising finding opens up new research directions for the LTSF task. We also advocate revisiting the validity of Transformer-based solutions for other time series analysis tasks (e.g., anomaly detection) in the future.

## In TSFLab
Default config: `configs/models/Linear.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

**Paper-driven local implementation.** The model implements the paper equation `X_hat_i = W X_i` as one temporal affine map shared across channels. The optional `individual` setting is a documented repository extension with one map per channel. The external repository is reference-only; no source file was copied or adapted. Paper preprocessing and reported benchmark values are outside code-validation scope.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/ZengCZ023,
  author       = {Ailing Zeng and
                  Muxi Chen and
                  Lei Zhang and
                  Qiang Xu},
  editor       = {Brian Williams and
                  Yiling Chen and
                  Jennifer Neville},
  title        = {Are Transformers Effective for Time Series Forecasting?},
  booktitle    = {Thirty-Seventh {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2023, Thirty-Fifth Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2023, Thirteenth Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2023, Washington, DC, USA, February
                  7-14, 2023},
  pages        = {11121--11128},
  publisher    = {{AAAI} Press},
  year         = {2023},
  url          = {https://doi.org/10.1609/aaai.v37i9.26317},
  doi          = {10.1609/AAAI.V37I9.26317},
  timestamp    = {Wed, 18 Mar 2026 17:07:12 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/ZengCZ023.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

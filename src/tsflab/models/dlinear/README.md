---
name: "DLinear"
summary: "DLinear is a time series forecasting model that decomposes the input sequence into a trend component and a seasonal (remainder) component and applies two independent one-layer linear projections to produce the final forecast. It serves as the primary model in the LTSF-Linear family and demonstrates that embarrassingly simple linear architectures can consistently outperform sophisticated Transformer-based long-term forecasters on standard benchmarks."
paper: "https://arxiv.org/abs/2205.13504"
paper_title: "Are Transformers Effective for Time Series Forecasting?"
venue: "AAAI 2023"
year: 2023
code: "https://github.com/cure-lab/LTSF-Linear"
revision: "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"
license: "Apache-2.0"
---
# DLinear

DLinear is a time series forecasting model that decomposes the input sequence into a trend component and a seasonal (remainder) component and applies two independent one-layer linear projections to produce the final forecast. It serves as the primary model in the LTSF-Linear family and demonstrates that embarrassingly simple linear architectures can consistently outperform sophisticated Transformer-based long-term forecasters on standard benchmarks.

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
[`configs/models/DLinear.toml`](../../../../configs/models/DLinear.toml).

## Differences

**Paper-driven local implementation.** A centered moving average separates trend and seasonal remainder; two temporal affine maps forecast the components and their outputs are summed. The implementation reuses the repository's validated `DLinearBackbone`, decomposition, and channel-wise projection contracts. Configurable kernel and per-channel projections are local extensions. The external repository is reference-only and no source file was copied or adapted.

## Shared components

- [`dlinear`](../_components/dlinear/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `kernel_size=25`, `individual=False`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Are Transformers Effective for Time Series Forecasting?
- **Venue**: AAAI 2023 (Oral)
- **Published**: 2023 (arXiv: 2022-05)
- **arXiv**: https://arxiv.org/abs/2205.13504

## Abstract
Recently, there has been a surge of Transformer-based solutions for the long-term time series forecasting (LTSF) task. Despite the growing performance over the past few years, we question the validity of this line of research in this work. Specifically, Transformers is arguably the most successful solution to extract the semantic correlations among the elements in a long sequence. However, in time series modeling, we are to extract the temporal relations in an ordered set of continuous points. While employing positional encoding and using tokens to embed sub-series in Transformers facilitate preserving some ordering information, the nature of the permutation-invariant self-attention mechanism inevitably results in temporal information loss. To validate our claim, we introduce a set of embarrassingly simple one-layer linear models named LTSF-Linear for comparison. Experimental results on nine real-life datasets show that LTSF-Linear surprisingly outperforms existing sophisticated Transformer-based LTSF models in all cases, and often by a large margin. Moreover, we conduct comprehensive empirical studies to explore the impacts of various design elements of LTSF models on their temporal relation extraction capability. We hope this surprising finding opens up new research directions for the LTSF task. We also advocate revisiting the validity of Transformer-based solutions for other time series analysis tasks (e.g., anomaly detection) in the future.

## In TSFLab
Default config: `configs/models/DLinear.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

**Paper-driven local implementation.** A centered moving average separates trend and seasonal remainder; two temporal affine maps forecast the components and their outputs are summed. The implementation reuses the repository's validated `DLinearBackbone`, decomposition, and channel-wise projection contracts. Configurable kernel and per-channel projections are local extensions. The external repository is reference-only and no source file was copied or adapted.

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

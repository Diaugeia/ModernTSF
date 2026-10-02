---
name: "TimeMosaic"
summary: "TimeMosaic is a time-series forecasting model designed to handle temporal heterogeneity in multivariate data. It employs adaptive patch embedding to dynamically adjust segmentation granularity based on local information density, and a segment-wise decoder that treats each prediction horizon as a related but distinct sub-task, adapting to horizon-specific difficulty rather than applying a single uniform decoder."
paper: "https://arxiv.org/abs/2509.19406"
paper_title: "TimeMosaic: Temporal Heterogeneity Guided Time Series Forecasting via Adaptive Granularity Patch and Segment-wise Decoding"
venue: "AAAI 2026"
year: 2026
code: "https://github.com/BenchCouncil/TimeMosaic"
revision: "214423b7f0b4653d04620814380a9301580285cc"
license: "NOASSERTION"
---
# TimeMosaic

TimeMosaic is a time-series forecasting model designed to handle temporal heterogeneity in multivariate data. It employs adaptive patch embedding to dynamically adjust segmentation granularity based on local information density, and a segment-wise decoder that treats each prediction horizon as a related but distinct sub-task, adapting to horizon-specific difficulty rather than applying a single uniform decoder.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2509.19406); title: TimeMosaic: Temporal Heterogeneity Guided Time Series Forecasting via Adaptive Granularity Patch and Segment-wise Decoding; venue/year: AAAI 2026 / 2026
- [codebase](https://github.com/BenchCouncil/TimeMosaic); revision: `214423b7f0b4653d04620814380a9301580285cc`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/TimeMosaic.toml`](../../../../configs/models/TimeMosaic.toml).

## Differences

Pinned source inspection: `models/TimeMosaic.py`, `models/Mosaic.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

This local implementation maps the paper's adaptive patch equations to a
soft region-wise mixture over configured granularities, repeat-aligns candidate
patch sequences, and applies segment-specific prompts only to attention keys and
values before separate forecast heads. It uses end-to-end local training rather
than a frozen foundation backbone, Gumbel-hard selection, budget regularization,
or the reported 321-billion-observation pre-training corpus. The reference-only
repository was inspected at the pinned revision; no external source code was copied.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `patch_sizes=[4, 8, 16]`, `num_segments=4`, `num_heads=4`, `dropout=0.1`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: TimeMosaic: Temporal Heterogeneity Guided Time Series Forecasting via Adaptive Granularity Patch and Segment-wise Decoding
- **Venue**: AAAI 2026
- **Published**: 2026 (arXiv: 2025-09)
- **arXiv**: https://arxiv.org/abs/2509.19406

## Abstract
Multivariate time series forecasting is essential in domains such as finance, transportation, climate, and energy. However, existing patch-based methods typically adopt fixed-length segmentation, overlooking the heterogeneity of local temporal dynamics and the decoding heterogeneity of forecasting. Such designs lose details in information-dense regions, introduce redundancy in stable segments, and fail to capture the distinct complexities of short-term and long-term horizons. We propose TimeMosaic, a forecasting framework that aims to address temporal heterogeneity. TimeMosaic employs adaptive patch embedding to dynamically adjust granularity according to local information density, balancing motif reuse with structural clarity while preserving temporal continuity. In addition, it introduces segment-wise decoding that treats each prediction horizon as a related subtask and adapts to horizon-specific difficulty and information requirements, rather than applying a single uniform decoder. Extensive evaluations on benchmark datasets demonstrate that TimeMosaic delivers consistent improvements over existing methods, and our model trained on the large-scale corpus with 321 billion observations achieves performance competitive with state-of-the-art TSFMs.

## Source and verification

Pinned source inspection: `models/TimeMosaic.py`, `models/Mosaic.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed.

This local implementation maps the paper's adaptive patch equations to a
soft region-wise mixture over configured granularities, repeat-aligns candidate
patch sequences, and applies segment-specific prompts only to attention keys and
values before separate forecast heads. It uses end-to-end local training rather
than a frozen foundation backbone, Gumbel-hard selection, budget regularization,
or the reported 321-billion-observation pre-training corpus. The reference-only
repository was inspected at the pinned revision; no external source code was copied.

## In TSFLab
Default config: `configs/models/TimeMosaic.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/DingFHWWYZ26,
  author       = {Kuiye Ding and
                  Fanda Fan and
                  Chunyi Hou and
                  Zheya Wang and
                  Lei Wang and
                  Zhengxin Yang and
                  Jianfeng Zhan},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {TimeMosaic: Temporal Heterogeneity Guided Time Series Forecasting
                  via Adaptive Granularity Patch and Segment-wise Decoding},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {20790--20798},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i25.39218},
  doi          = {10.1609/AAAI.V40I25.39218},
  timestamp    = {Fri, 27 Mar 2026 17:13:39 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/DingFHWWYZ26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

---
name: "FTP"
summary: "FTP is a clean-room pure-MLP FusionTimePatch forecaster combining recursive channel-independent/channel-mixed patch views, Channel Enhancement, and linear fusion."
paper: "https://doi.org/10.1609/aaai.v40i33.40072"
paper_title: "Unifying Channel Independence and Mixing: Multi-Scale Patch Recursion for Global-Local Representation Synergy in Multivariate Time Series Forecasting"
venue: "AAAI 2026"
year: 2026
code: "https://github.com/Zhveh7/FTP"
revision: "964b6f614a1294f136d03049ee67b35f68605422"
license: "MIT"
---
# FTP

FTP (FusionTimePatch) is a pure-MLP multivariate forecaster that unifies channel-independent and channel-mixed processing across multiple temporal patch spans.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://doi.org/10.1609/aaai.v40i33.40072); title: Unifying Channel Independence and Mixing: Multi-Scale Patch Recursion for Global-Local Representation Synergy in Multivariate Time Series Forecasting; venue/year: AAAI 2026 / 2026
- [codebase](https://github.com/Zhveh7/FTP); revision: `964b6f614a1294f136d03049ee67b35f68605422`; license: `MIT`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/FTP.toml`](../../../../configs/models/FTP.toml).

## Differences

Pinned source inspection: `models/FTP.py`, `layers/FTP_EncDec.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed. Reference source code was inspected at the pinned revision; no external source code was copied. The rewrite follows Algorithm 1 and the published method: recursive
multiscale GLF-CI/GLF-CM branches, latent/channel scoring in CE, tri-stream
linear fusion, original-embedding concatenation, and an MLP horizon head.

The paper samples dominant channels probabilistically; this point forecaster
uses the probability-weighted expectation for deterministic execution. The
preset is compact and does not claim dataset-specific tuned widths or depths.
Evidence is in `../../../verification/evidence/FTP.json`.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `num_layers=2`, `patch_unit=4`, `num_scales=3`, `stride=2`, `dropout=0.1`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Unifying Channel Independence and Mixing: Multi-Scale Patch Recursion for Global-Local Representation Synergy in Multivariate Time Series Forecasting
- **Venue**: AAAI 2026
- **Published**: 2026
- **DOI**: https://doi.org/10.1609/aaai.v40i33.40072

## Abstract
The published AAAI paper introduces three core components: Dual-GLF, which introduces channel-independent (CI) and channel-mixing (CM) perspectives in parallel, leveraging multi-scale patch recursion to capture both local and global temporal patterns; a Channel Enhancement (CE) module that enhances salient channel features and diffuses them across channels, improving sensitivity to anomalies and underlying drivers; and linear fusion that aggregates the three complementary streams into a rich global-local representation. The model achieves competitive performance on standard long-term forecasting benchmarks. The official repository remains recorded above only as a reference-only provenance link.

## Source and verification

Pinned source inspection: `models/FTP.py`, `layers/FTP_EncDec.py` were examined at the recorded revision to confirm implementation details. The local module was written for TSFLab; no external source file is copied.

Local implementation: confirmed. Reference source code was inspected at the pinned revision; no external source code was copied. The rewrite follows Algorithm 1 and the published method: recursive
multiscale GLF-CI/GLF-CM branches, latent/channel scoring in CE, tri-stream
linear fusion, original-embedding concatenation, and an MLP horizon head.

The paper samples dominant channels probabilistically; this point forecaster
uses the probability-weighted expectation for deterministic execution. The
preset is compact and does not claim dataset-specific tuned widths or depths.
Evidence is in `../../../verification/evidence/FTP.json`.

## In TSFLab
Default config: `configs/models/FTP.toml`; model specification: `spec.py`; local implementation: `model.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/ZhangZBZGJSP26,
  author       = {Wenhao Zhang and
                  Chun Zhang and
                  Wei Bai and
                  Ning Zhang and
                  Changxia Gao and
                  Yuxin Jia and
                  Chenhao Shi and
                  Shaoxiong Pang},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {Unifying Channel Independence and Mixing: Multi-Scale Patch Recursion
                  for Global-Local Representation Synergy in Multivariate Time Series
                  Forecasting},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {28427--28436},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i33.40072},
  doi          = {10.1609/AAAI.V40I33.40072},
  timestamp    = {Tue, 24 Mar 2026 17:03:54 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/ZhangZBZGJSP26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

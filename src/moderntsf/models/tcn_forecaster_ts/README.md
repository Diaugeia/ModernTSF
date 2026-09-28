---
name: "TCNForecasterTS"
summary: "TCNForecasterTS is a clean-room temporal convolutional baseline with exponentially dilated causal residual blocks and a direct multistep forecast head."
paper: "https://arxiv.org/abs/1803.01271"
paper_title: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
venue: "arXiv preprint"
year: 2018
---
# TCNForecasterTS

TCNForecasterTS is a clean-room temporal convolutional baseline with exponentially dilated causal residual blocks and a direct multistep forecast head.

<!-- model-card:canonical:start -->
## Method overview

TCNForecasterTS is a clean-room temporal convolutional baseline with exponentially dilated causal residual blocks and a direct multistep forecast head.

## Core architecture

TCNForecasterTS is a clean-room temporal convolutional baseline with exponentially dilated causal residual blocks and a direct multistep forecast head.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/1803.01271); title: An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling; venue/year: arXiv preprint / 2018
- codebase: not available

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/TCNForecasterTS.toml`](../../../../configs/models/TCNForecasterTS.toml).

## Differences

Clean-room implementation: confirmed. The local code was independently designed from the causal, dilated residual architecture described by Bai et al.; no external implementation source was copied. It omits paper-side weight normalization, uses a final-timestep direct horizon head, and optionally applies RevIN, so no paper-result reference comparison is claimed. Causality and full runtime-contract evidence are recorded in `../../../verification/evidence/TCNForecasterTS.json`.

## Shared components

- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_layers=2`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
- **Venue**: arXiv preprint
- **Published**: 2018
- **arXiv**: https://arxiv.org/abs/1803.01271

## Abstract
For most deep learning practitioners, sequence modeling is synonymous with recurrent networks. Yet recent results indicate that convolutional architectures can outperform recurrent networks on tasks such as audio synthesis and machine translation. Given a new sequence modeling task or dataset, which architecture should one use? We conduct a systematic evaluation of generic convolutional and recurrent architectures for sequence modeling. The models are evaluated across a broad range of standard tasks that are commonly used to benchmark recurrent networks. Our results indicate that a simple convolutional architecture outperforms canonical recurrent networks such as LSTMs across a diverse range of tasks and datasets, while demonstrating longer effective memory. We conclude that the common association between sequence modeling and recurrent networks should be reconsidered, and convolutional networks should be regarded as a natural starting point for sequence modeling tasks. To assist related work, we have made code available at http://github.com/locuslab/TCN.

## Source and verification

Clean-room implementation: confirmed. The local code was independently designed from the causal, dilated residual architecture described by Bai et al.; no external implementation source was copied. It omits paper-side weight normalization, uses a final-timestep direct horizon head, and optionally applies RevIN, so no paper-result reference comparison is claimed. Causality and full runtime-contract evidence are recorded in `../../../verification/evidence/TCNForecasterTS.json`.

## In ModernTSF
Default config: `configs/models/TCNForecasterTS.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-1803-01271,
  author       = {Shaojie Bai and
                  J. Zico Kolter and
                  Vladlen Koltun},
  title        = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks
                  for Sequence Modeling},
  journal      = {CoRR},
  volume       = {abs/1803.01271},
  year         = {2018},
  url          = {http://arxiv.org/abs/1803.01271},
  eprinttype   = {arXiv},
  eprint       = {1803.01271},
  timestamp    = {Mon, 13 Aug 2018 16:47:39 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1803-01271.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

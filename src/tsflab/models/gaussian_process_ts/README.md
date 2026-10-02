---
name: "GaussianProcessTS"
summary: "GaussianProcessTS is a sparse RBF-kernel posterior-mean approximation using learned inducing inputs and horizon targets for channel-wise lag forecasting."
paper: "https://gaussianprocess.org/gpml/chapters/"
paper_title: "Gaussian Processes for Machine Learning"
venue: "MIT Press"
year: 2006
---
# GaussianProcessTS

GaussianProcessTS is a sparse RBF-kernel posterior-mean approximation using learned inducing inputs and horizon targets for channel-wise lag forecasting.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://gaussianprocess.org/gpml/chapters/); title: Gaussian Processes for Machine Learning; venue/year: MIT Press / 2006
- codebase: not available

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/GaussianProcessTS.toml`](../../../../configs/models/GaussianProcessTS.toml).

## Differences

This is an independent inducing-basis mean approximation, not exact GP regression: inducing pairs are learned by gradient descent, channels share one function, and posterior covariance or calibrated uncertainty is not returned. It is not equivalent to any third-party GP package, and no such source was inspected or copied.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `num_inducing=16`, `length_scale=1.0`, `noise=0.001`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Gaussian Processes for Machine Learning
- **Venue**: MIT Press
- **Published**: 2006
- **Link**: https://gaussianprocess.org/gpml/chapters/

## Abstract
Gaussian Process regression places a prior over functions and conditions kernel values on observations. The local approximation learns inducing lag/forecast pairs and evaluates an RBF kernel linear solve for the posterior mean only; it does not return posterior covariance.

## In TSFLab
Default config: `configs/models/GaussianProcessTS.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Source and verification

This is an independent inducing-basis mean approximation, not exact GP regression: inducing pairs are learned by gradient descent, channels share one function, and posterior covariance or calibrated uncertainty is not returned. It is not equivalent to any third-party GP package, and no such source was inspected or copied.

## Citation

```bibtex
@book{DBLP:books/lib/RasmussenW06,
  author       = {Carl Edward Rasmussen and
                  Christopher K. I. Williams},
  title        = {Gaussian processes for machine learning},
  series       = {Adaptive computation and machine learning},
  publisher    = {{MIT} Press},
  year         = {2006},
  url          = {https://www.worldcat.org/oclc/61285753},
  isbn         = {026218253X},
  timestamp    = {Fri, 17 Jul 2020 16:12:42 +0200},
  biburl       = {https://dblp.org/rec/books/lib/RasmussenW06.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

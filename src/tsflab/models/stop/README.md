---
name: "STOP"
summary: "STOP (Spatio-Temporal OOD Processor) is a spatiotemporal forecasting model that addresses out-of-distribution generalization in graph-structured data by replacing node-to-node message passing with a centralized messaging mechanism using Context-Aware Units, combined with a message perturbation mechanism and distributionally robust optimization to produce forecasts that generalize across spatial and temporal distribution shifts."
paper: "https://proceedings.mlr.press/v267/ma25s.html"
paper_title: "Robust Spatio-Temporal Centralized Interaction for OOD Learning"
venue: "ICML 2025"
year: 2025
code: "https://github.com/PoorOtterBob/STOP"
revision: "8babb610ece36a4215b2f66e1ef4a154f0c4f440"
license: "NOASSERTION"
tagline: "Centralized messaging through a few learnable context units over trend/residual encodings and calendar prompts."
tags: ["mlp", "attention-variant", "spatiotemporal", "decomposition", "covariates", "channel-mixing"]
composition: ["normalization=none", "decomposition=component:series_decomposition", "temporal=local:long-short-linear-encoders+local:residual-mlp-mixer", "channel=local:centralized-context-unit-interaction", "head=local:dual-linear-temporal-spatial-heads", "loss=loss:mse"]
---
# STOP

## Key ideas

- `SeriesDecomposition` splits each node's lookback; the trend and residual are encoded by separate linear maps and concatenated with time-of-day and day-of-week prompt embeddings.
- `CentralizedInteraction` replaces node-to-node message passing: nodes aggregate into a small bank of learnable context units (`core`) and diffuse back, giving a context and a personalized remainder.
- A refine MLP and a second residual mixer produce a spatial branch; two linear heads (temporal and spatial) are summed for the forecast.
- `environment_forecasts` produces perturbed-message forecasts for the paper's robust-optimization idea, but it is not called by `forward`.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://proceedings.mlr.press/v267/ma25s.html); title: Robust Spatio-Temporal Centralized Interaction for OOD Learning; venue/year: ICML 2025 / 2025
- [codebase](https://github.com/PoorOtterBob/STOP); revision: `8babb610ece36a4215b2f66e1ef4a154f0c4f440`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/STOP.toml`](../../../../configs/models/STOP.toml).

## Differences

Clean-room implementation: confirmed. The reference-only source code was not
copied. The structure map covers equations (4)--(20), including ConAU
centralized interaction and bounded GenPU-style environment forecasts; the
external experiment loop remains responsible for worst-loss DRO selection.

## Shared components

- [`marks`](../_components/marks/README.md)
- [`series_decomposition`](../_components/series_decomposition/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=6`, `model_dim=16`, `prompt_dim=16`, `num_layer=2`, `hid_dim=64`, `tod_size=24`, `kernel_size=3`, `core=4`, `head=4`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Robust Spatio-Temporal Centralized Interaction for OOD Learning
- **Venue**: ICML 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
Recently, spatiotemporal graph convolutional networks have achieved dominant performance in spatiotemporal prediction tasks. However, most models relying on node-to-node messaging interaction exhibit sensitivity to spatiotemporal shifts, encountering out-of-distribution (OOD) challenges. To address these issues, we introduce Spatio-Temporal OOD Processor (STOP), which employs a centralized messaging mechanism along with a message perturbation mechanism to facilitate robust spatiotemporal interactions. Specifically, the centralized messaging mechanism integrates Context-Aware Units for coarse-grained spatiotemporal feature interactions with nodes, effectively blocking traditional node-to-node messages. We also implement a message perturbation mechanism to disrupt this messaging process, compelling the model to extract generalizable contextual features from generated variant environments. Finally, we customize a spatiotemporal distributionally robust optimization approach that exposes the model to challenging environments, thereby further enhancing its generalization capabilities. Compared with 14 baselines across six datasets, STOP achieves up to 17.01% improvement in generalization performance and 18.44% improvement in inductive learning performance.

## In TSFLab
Default config: `configs/models/STOP.toml`; model specification: `spec.py`; implementation: `model.py`.

The local implementation is paper-derived and never passes messages directly
between graph nodes: all spatial interaction goes through a small ConAU bank.
The unlicensed author repository is retained only as provenance and its source
was not copied. OOD splits and alternating GenPU/DRO optimization are not part
of the generic trainer.

## Verification

Clean-room implementation: confirmed. The reference-only source code was not
copied. The structure map covers equations (4)--(20), including ConAU
centralized interaction and bounded GenPU-style environment forecasts; the
external experiment loop remains responsible for worst-loss DRO selection.

## Citation

```bibtex
@inproceedings{DBLP:conf/icml/MaW0Z0025,
  author       = {Jiaming Ma and
                  Binwu Wang and
                  Pengkun Wang and
                  Zhengyang Zhou and
                  Xu Wang and
                  Yang Wang},
  editor       = {Aarti Singh and
                  Maryam Fazel and
                  Daniel Hsu and
                  Simon Lacoste{-}Julien and
                  Felix Berkenkamp and
                  Tegan Maharaj and
                  Kiri Wagstaff and
                  Jerry Zhu},
  title        = {Robust Spatio-Temporal Centralized Interaction for {OOD} Learning},
  booktitle    = {Forty-second International Conference on Machine Learning, {ICML}
                  2025, Vancouver, BC, Canada, July 13-19, 2025},
  series       = {Proceedings of Machine Learning Research},
  publisher    = {{PMLR} / OpenReview.net},
  year         = {2025},
  url          = {https://proceedings.mlr.press/v267/ma25s.html},
  timestamp    = {Wed, 04 Feb 2026 17:22:46 +0100},
  biburl       = {https://dblp.org/rec/conf/icml/MaW0Z0025.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

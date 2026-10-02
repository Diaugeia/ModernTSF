---
name: "BiST"
summary: "BiST is a spatiotemporal learning model for node-structured or graph-structured data that simultaneously captures temporal dynamics and spatial relationships between nodes. It challenges the standard input-label spatiotemporal consistency assumption by incorporating label information during training via a lightweight bidirectional MLP backbone with an adaptive graph, enabling strong predictive performance with a fraction of the training time and memory of existing methods."
paper: "https://www.vldb.org/pvldb/vol18/p1663-wang.pdf"
paper_title: "BiST: A Lightweight and Efficient Bi-Directional Model for Spatiotemporal Prediction"
venue: "PVLDB 2025"
year: 2025
code: "https://github.com/PoorOtterBob/BiST"
revision: "dd94adf7721fcbb9e3feb5d1b44040305199a4cc"
license: "NOASSERTION"
tagline: "MLP forecaster with label-side residual correction: virtual-cluster context and adaptive diffusion refine a base."
tags: ["mlp", "decomposition", "spatiotemporal", "graph-learning", "covariates", "lightweight"]
composition: ["normalization=none", "decomposition=component:series_decomposition", "temporal=local:residual-mlp-forward-and-backward-representations", "channel=local:virtual-cluster-context-graph+local:adaptive-diffusion-kernel", "head=local:linear-base-and-correction-heads", "loss=loss:mse"]
---
# BiST

## Key ideas

- `series_decomposition` splits stable and trend parts that are projected by separate linear maps, then concatenated with learned node, time-of-day and weekday prompts.
- A stack of `ResidualMLP` layers makes the forward representation, and `base_head` produces the base forecast.
- `node_queries` and `cluster_keys` define soft virtual clusters; their context graph separates common and personalized parts, from which a residual branch is built.
- `_adaptive_kernel` diffuses the residual over a learned graph with node-wise alpha/beta; `correction_head` adds the result to the base forecast.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://www.vldb.org/pvldb/vol18/p1663-wang.pdf); title: BiST: A Lightweight and Efficient Bi-Directional Model for Spatiotemporal Prediction; venue/year: PVLDB 2025 / 2025
- [codebase](https://github.com/PoorOtterBob/BiST); revision: `dd94adf7721fcbb9e3feb5d1b44040305199a4cc`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/BiST.toml`](../../../../configs/models/BiST.toml).

## Differences

**Clean-room implementation: confirmed.** Equations 5--24 are mapped to the
local decomposition, prompt, forward representation, virtual-cluster residual,
adaptive diffusion, and correction modules. No author-source implementation was
copied. Published-metric and checkpoint reference comparison are not claimed.

## Shared components

- [`marks`](../_components/marks/README.md)
- [`series_decomposition`](../_components/series_decomposition/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=6`, `model_dim=32`, `prompt_dim=16`, `num_layers=3`, `tod_size=24`, `kernel_size=3`, `residual_steps=2`, `graph_dim=8`, `virtual_clusters=8`
<!-- model-card:canonical:end -->

## Paper
- **Title**: BiST: A Lightweight and Efficient Bi-Directional Model for Spatiotemporal Prediction
- **Venue**: Proceedings of the VLDB Endowment (PVLDB), Vol. 18, No. 6
- **Published**: 2025
- **arXiv**: N/A

## Abstract
While existing spatiotemporal prediction models have shown promising performance, they often rely on the assumption of input-label spatiotemporal consistency, and their high complexity raises concerns about scalability. BiST addresses these issues by decomposing the prediction into a forward spatiotemporal learning process that generates base predictions and a residual correction process that models spatiotemporal residuals to refine those predictions. The backbone is a lightweight MLP rather than stacked spatiotemporal layers, yielding competitive accuracy while consuming only a small fraction of the training time and memory of state-of-the-art models.

## In TSFLab
Default config: `configs/models/BiST.toml`; model specification: `spec.py`;
clean-room implementation: `model.py`. The unlicensed repository remains a
paper-discovery reference only.

## Verification

**Clean-room implementation: confirmed.** Equations 5--24 are mapped to the
local decomposition, prompt, forward representation, virtual-cluster residual,
adaptive diffusion, and correction modules. No author-source implementation was
copied. Published-metric and checkpoint reference comparison are not claimed.

## Citation

```bibtex
@article{DBLP:journals/pvldb/MaWWZWW25,
  author       = {Jiaming Ma and
                  Binwu Wang and
                  Pengkun Wang and
                  Zhengyang Zhou and
                  Xu Wang and
                  Yang Wang},
  title        = {BiST: {A} Lightweight and Efficient Bi-directional Model for Spatiotemporal
                  Prediction},
  journal      = {Proc. {VLDB} Endow.},
  volume       = {18},
  number       = {6},
  pages        = {1663--1676},
  year         = {2025},
  url          = {https://www.vldb.org/pvldb/vol18/p1663-wang.pdf},
  doi          = {10.14778/3725688.3725697},
  timestamp    = {Wed, 17 Dec 2025 16:44:24 +0100},
  biburl       = {https://dblp.org/rec/journals/pvldb/MaWWZWW25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

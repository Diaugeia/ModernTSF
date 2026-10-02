---
name: "MAGE"
summary: "MAGE (Mixture of Adaptive Graph Experts) is a spatiotemporal learning model for node-structured or graph-structured data. It introduces a sparse yet balanced mixture-of-experts strategy in which each expert perceives a unique underlying graph topology through kernel-based functions with linear complexity relative to the number of nodes, overcoming the noise amplification caused by ReLU activations in existing adaptive graph learning methods."
paper: "https://proceedings.neurips.cc/paper_files/paper/2025/hash/54c9bfb0885ae07f23607f617ab64c2b-Abstract-Conference.html"
paper_title: "Less but More: Linear Adaptive Graph Learning Empowering Spatiotemporal Forecasting"
venue: "NeurIPS 2025"
year: 2025
code: "https://github.com/PoorOtterBob/MAGE"
revision: "f1fdd27da4e72a140c4f341f94d368fbcaec7507"
license: "NOASSERTION"
---
# MAGE

MAGE (Mixture of Adaptive Graph Experts) is a spatiotemporal learning model for node-structured or graph-structured data. It introduces a sparse yet balanced mixture-of-experts strategy in which each expert perceives a unique underlying graph topology through kernel-based functions with linear complexity relative to the number of nodes, overcoming the noise amplification caused by ReLU activations in existing adaptive graph learning methods.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/54c9bfb0885ae07f23607f617ab64c2b-Abstract-Conference.html); title: Less but More: Linear Adaptive Graph Learning Empowering Spatiotemporal Forecasting; venue/year: NeurIPS 2025 / 2025
- [codebase](https://github.com/PoorOtterBob/MAGE); revision: `f1fdd27da4e72a140c4f341f94d368fbcaec7507`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/MAGE.toml`](../../../../configs/models/MAGE.toml).

## Differences

Clean-room implementation: confirmed. The reference-only source code was not
copied. The structure map covers factorised graph kernels, sparse balanced
routing, and three recurrent depths. Calendar prompting is reduced and the
training expert-count objective is omitted.

`MixtureGraphBlock`'s gate was compared against the `topk_expert_router`
component: it selects top-k on raw logits then applies a masked softmax
(sums to one only over selected experts) and blends 5%/95% with the
batch/node-mean dense softmax, which is a different routing formula from the
floor-renormalized `topk_dense_mix` used by DUET and DynamicTMoE. It stays
model-local rather than forcing a flag-driven shared abstraction.

## Shared components

- [`marks`](../_components/marks/README.md)

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=6`, `model_dim=64`, `recur_num=8`, `topk=2`, `node_dim=16`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Less but More: Linear Adaptive Graph Learning Empowering Spatiotemporal Forecasting
- **Venue**: NeurIPS 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
The effectiveness of Spatiotemporal Graph Neural Networks (STGNNs) critically hinges on the quality of the underlying graph topology. While end-to-end adaptive graph learning methods have demonstrated promising results in capturing latent spatiotemporal dependencies, they often suffer from high computational complexity and limited expressive capacity. In this paper, we propose MAGE for efficient spatiotemporal forecasting. We first conduct a theoretical analysis demonstrating that the ReLU activation function employed in existing methods amplifies edge-level noise during graph topology learning, thereby compromising the fidelity of the learned graph structures. To enhance model expressiveness, we introduce a sparse yet balanced mixture-of-experts strategy, where each expert perceives the unique underlying graph through kernel-based functions and operates with linear complexity relative to the number of nodes. The sparsity mechanism ensures that each node interacts exclusively with compatible experts, while the balancing mechanism promotes uniform activation across all experts, enabling diverse and adaptive graph representations. Furthermore, we theoretically establish that a single graph convolution using the learned graph in MAGE is mathematically equivalent to multiple convolutional steps under conventional graphs. We evaluate MAGE against advanced baselines on multiple real-world spatiotemporal datasets. MAGE achieves competitive performance while maintaining strong computational efficiency.

## In TSFLab
Default config: `configs/models/MAGE.toml`; model specification: `spec.py`; implementation: `model.py`.

The local module is independently implemented from the NeurIPS paper. Each
expert performs node→basis→node kernel propagation in linear node complexity;
top-k routing is combined with a small balancing path, and three recurrent
depths feed the residual forecast. The unlicensed author repository remains a
reference-only link and none of its source is included.

## Verification

Clean-room implementation: confirmed. The reference-only source code was not
copied. The structure map covers factorised graph kernels, sparse balanced
routing, and three recurrent depths. Calendar prompting is reduced and the
training expert-count objective is omitted.

`MixtureGraphBlock`'s gate was compared against the `topk_expert_router`
component: it selects top-k on raw logits then applies a masked softmax
(sums to one only over selected experts) and blends 5%/95% with the
batch/node-mean dense softmax, which is a different routing formula from the
floor-renormalized `topk_dense_mix` used by DUET and DynamicTMoE. It stays
model-local rather than forcing a flag-driven shared abstraction.

## Citation

```bibtex
@inproceedings{ma2025less,
  author    = {Jiaming Ma and Binwu Wang and Guanjun Wang and Kuo Yang and Zhengyang Zhou and Pengkun Wang and Xu Wang and Yang Wang},
  title     = {Less but More: Linear Adaptive Graph Learning Empowering Spatiotemporal Forecasting},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  url       = {https://github.com/PoorOtterBob/MAGE}
}
```

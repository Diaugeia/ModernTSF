---
name: "AdaMSHyper"
summary: "Ada-MSHyper builds a strided-convolution multi-scale pyramid of the normalized series, learns a sparse binary node-to-hyperedge incidence matrix at every scale from node and hyperedge embeddings, regularizes it with node and hyperedge constraint losses, updates nodes with intra-scale hypergraph convolution attention, mixes all scales' hyperedges with self-attention, and predicts with one linear layer over the concatenated node and hyperedge tokens."
paper: "https://arxiv.org/abs/2410.23992"
paper_title: "Ada-MSHyper: Adaptive Multi-Scale Hypergraph Transformer for Time Series Forecasting"
venue: "NeurIPS 2024"
year: 2024
code: "https://github.com/shangzongjiang/Ada-MSHyper"
revision: "8efd34c8737b0e4dafbd247794213d0278e1dc06"
license: "NOASSERTION"
tagline: "Per-scale learned hypergraphs over a conv pyramid, node/hyperedge constraint loss, intra-scale and hyperedge attention."
tags: ["gnn", "hypergraph", "multi-scale", "graph-learning", "attention-variant", "normalization", "revin"]
composition: ["normalization=component:revin", "decomposition=local:strided-conv-multi-scale-pyramid", "temporal=component:adaptive_node_embedding_adjacency+local:intra-scale-hypergraph-attention-and-inter-scale-hyperedge-attention", "channel=local:channel-mixing-feature-projections", "head=local:concat-node-hyperedge-linear-head", "loss=loss:mse+local:node-hyperedge-constraint"]
---
# AdaMSHyper

## Key ideas

- `Model.multi_scale` aggregates the series with strided `ScaleConv` layers (Conv1d, BatchNorm, ELU; window equals stride), giving node sets of length `L`, `L/w1`, `L/(w1 w2)`.
- `ScaleHypergraph.incidence` learns an incidence matrix per scale: `adaptive_node_embedding_adjacency` (`softmax(relu(E_node E_hyper^T))`), top-`eta` per node, threshold `beta`, binary in the forward value.
- Hyperedge features are the mean of member nodes; the node constraint (distance of a node to its hyperedges) and the hyperedge constraint (cosine-weighted attract/repel by Euclidean distance with margin `gamma`) form `L_const`, exposed as `aux_loss` while training.
- Nodes are updated by attention-enriched, degree-normalized hypergraph convolution; all scales' hyperedges then exchange information by one self-attention (`hyperedge_attention`), and a single Linear maps the concatenated node and hyperedge tokens to the horizon.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2410.23992); title: Ada-MSHyper: Adaptive Multi-Scale Hypergraph Transformer for Time Series Forecasting; venue/year: NeurIPS 2024 / 2024
- [codebase](https://github.com/shangzongjiang/Ada-MSHyper); revision: `8efd34c8737b0e4dafbd247794213d0278e1dc06`; license: `NOASSERTION`

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/AdaMSHyper.toml`](../../../../configs/models/AdaMSHyper.toml).

## Differences

Paper and official code (pinned revision `8efd34c`, `models/ASHyper.py`) were consulted for structure only; the official repository has no license file and nothing was copied. The local module follows the paper's equations (Eq. 3-16). The official code departs from the paper in several places, and the local module follows the paper. Recorded differences:

- Prediction head: the official forward sums three branches (a DLinear-style `seq_len -> pred_len` linear on the normalized input, a linear over the hypergraph-convolved node tokens initialized to the average, and a linear over hyperedge-attention tokens zero-padded to a hard-coded 80) and then applies another `pred_len -> pred_len` linear. The local model uses the paper's single linear over the concatenated node tokens of all scales and the attended hyperedge tokens, with no input skip branch.
- Incidence learning: the official adjacency is made binary with `torch.where`, which blocks gradients to the node and hyperedge embeddings, scales logits by a fixed `alpha=3` (`softmax(relu(3 E_node E_hyper^T))`), and drops empty hyperedge columns. The local incidence uses `softmax(relu(E_node E_hyper^T))` without the factor 3, adds a straight-through term so the soft scores receive gradients, and keeps empty hyperedges masked out of attention, constraints and the output.
- Constraint losses: the official node loss is `|mean(v_i - e_j)|` over member pairs and its hyperedge loss sums over all ordered pairs (including a hyperedge with itself) with a hard-coded margin 4.2 and `|mean|` per pair, combined with unit weights, its validation loss weights `0.2 * MSE + 0.8 * |constraint|`, and its training loop keeps a second optimizer object for the constraint term. The local node loss is the mean absolute feature gap of a node to its hyperedges and the hyperedge loss averages over non-empty pairs (Eq. 10-12); they are combined as `const_weight * (lambda_balance * node + (1 - lambda_balance) * hyperedge)` (Eq. 13) and exposed as `aux_loss` during training only, added to the MSE criterion by the shared trainer.
- Multi-scale construction: local `ScaleConv` is a strided Conv1d + BatchNorm + ELU per scale with window = stride and no final LayerNorm; the official default CSCM is `Bottleneck_Construct` (Linear down/up projection around the convolutions, three convolution layers in `Conv_Construct`, final LayerNorm). Default windows are `[4, 4]` (three scales) and `hyper_num = [50, 20, 10]`, `eta` corresponds to the official `k = 3`.
- The hypergraph feature width is the channel count (`enc_in`) rather than a `d_model`-wide projection; `d_embed` is the node and hyperedge embedding width (official `d_model`). The official per-scale `linhy`/`linnod` linear layers are unused in its forward and omitted. The official hard-coded sizes (80, 76, 100, 320) are replaced by sizes derived from `seq_len` and `hyper_num`.
- Normalization is the shared `revin` component without affine parameters, equal to the official mean/std normalization (`+1e-5`).

## Shared components

- [`adaptive_node_embedding_adjacency`](../_components/adaptive_node_embedding_adjacency/README.md)
- [`revin`](../_components/revin/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `window_size=[4, 4]`, `hyper_num=[50, 20, 10]`, `d_embed=16`, `eta=3`, `beta=0.5`, `gamma=4.2`, `lambda_balance=0.5`, `const_weight=1.0`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Source and verification

Paper and official code (pinned revision `8efd34c`, `models/ASHyper.py`) were consulted for structure only; the official repository has no license file and nothing was copied. The local module follows the paper's equations (Eq. 3-16). The official code departs from the paper in several places, and the local module follows the paper. Recorded differences:

- Prediction head: the official forward sums three branches (a DLinear-style `seq_len -> pred_len` linear on the normalized input, a linear over the hypergraph-convolved node tokens initialized to the average, and a linear over hyperedge-attention tokens zero-padded to a hard-coded 80) and then applies another `pred_len -> pred_len` linear. The local model uses the paper's single linear over the concatenated node tokens of all scales and the attended hyperedge tokens, with no input skip branch.
- Incidence learning: the official adjacency is made binary with `torch.where`, which blocks gradients to the node and hyperedge embeddings, scales logits by a fixed `alpha=3` (`softmax(relu(3 E_node E_hyper^T))`), and drops empty hyperedge columns. The local incidence uses `softmax(relu(E_node E_hyper^T))` without the factor 3, adds a straight-through term so the soft scores receive gradients, and keeps empty hyperedges masked out of attention, constraints and the output.
- Constraint losses: the official node loss is `|mean(v_i - e_j)|` over member pairs and its hyperedge loss sums over all ordered pairs (including a hyperedge with itself) with a hard-coded margin 4.2 and `|mean|` per pair, combined with unit weights, its validation loss weights `0.2 * MSE + 0.8 * |constraint|`, and its training loop keeps a second optimizer object for the constraint term. The local node loss is the mean absolute feature gap of a node to its hyperedges and the hyperedge loss averages over non-empty pairs (Eq. 10-12); they are combined as `const_weight * (lambda_balance * node + (1 - lambda_balance) * hyperedge)` (Eq. 13) and exposed as `aux_loss` during training only, added to the MSE criterion by the shared trainer.
- Multi-scale construction: local `ScaleConv` is a strided Conv1d + BatchNorm + ELU per scale with window = stride and no final LayerNorm; the official default CSCM is `Bottleneck_Construct` (Linear down/up projection around the convolutions, three convolution layers in `Conv_Construct`, final LayerNorm). Default windows are `[4, 4]` (three scales) and `hyper_num = [50, 20, 10]`, `eta` corresponds to the official `k = 3`.
- The hypergraph feature width is the channel count (`enc_in`) rather than a `d_model`-wide projection; `d_embed` is the node and hyperedge embedding width (official `d_model`). The official per-scale `linhy`/`linnod` linear layers are unused in its forward and omitted. The official hard-coded sizes (80, 76, 100, 320) are replaced by sizes derived from `seq_len` and `hyper_num`.
- Normalization is the shared `revin` component without affine parameters, equal to the official mean/std normalization (`+1e-5`).

## Citation

```bibtex
@inproceedings{shang2024adamshyper,
  title     = {{Ada-MSHyper}: Adaptive Multi-Scale Hypergraph Transformer for Time Series Forecasting},
  author    = {Shang, Zongjiang and Chen, Ling and Wu, Binqing and Cui, Dongliang},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```

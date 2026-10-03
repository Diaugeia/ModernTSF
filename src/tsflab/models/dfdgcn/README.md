---
name: "DFDGCN"
description: "Traffic GNN: gated dilated temporal convolutions with graph propagation over static, adaptive and a per-sample frequency-domain graph built from FFT magnitudes and node identity. Use for sensor networks with a road graph and calendar marks; not for data without node structure."
---

# DFDGCN

## Idea

- `FrequencyGraph` builds a directed per-sample graph from the FFT magnitude of each node's history plus a node identity embedding, reducing time-shift sensitivity.
- `DynamicGraphMix` propagates features over four graphs (forward/reverse supports, a self-adaptive graph, and the frequency graph) and projects the stacked hops.
- The temporal backbone stacks `gated_dilated_conv` layers with residual and skip connections; a conv head outputs all horizons.
- Calendar marks give time-of-day and day-of-week input channels.

## When to use

- Traffic or sensor networks where node patterns are similar but shifted in time, so frequency magnitudes reveal relations that raw values hide.
- Needs a node graph (`adj_mx`) and benefits from daily/weekly calendar marks; point output only.

## Configure

- `enc_in`: number of graph nodes N.
- `adj_mx`: the dataset's `[N, N]` adjacency.

Other hyperparameters: preset defaults in `configs/models/DFDGCN.toml`; tune generically.

## Differences

- Local rewrite reconstructing the dilated backbone, predefined/adaptive/dynamic graph mixture, frequency graph, embeddings and head from the MIT official code at `31050585`.
- The preset uses smaller widths and two blocks instead of the official four.
- `a` and `subgraph` are accepted for config compatibility but unused by the model.
- Official preprocessing, masked-MAE training, and published numerical results are not included.

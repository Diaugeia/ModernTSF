---
name: "Gateformer"
summary: "Gateformer encodes each variate through two complementary pathways: a whole-window global embedding and a patched cross-time-attention embedding. It fuses them with a learned gate, applies cross-variate attention on the fused variate embeddings, and fuses again with a second gate before projecting to the forecast horizon."
paper: "https://arxiv.org/abs/2505.00307"
paper_title: "Gateformer: Advancing Multivariate Time Series Forecasting through Temporal and Variate-Wise Attention with Gated Representations"
venue: "arXiv preprint"
year: 2025

---
# Gateformer

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2505.00307); title: Gateformer: Advancing Multivariate Time Series Forecasting through Temporal and Variate-Wise Attention with Gated Representations; venue/year: arXiv preprint / 2025
- codebase: not available

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/Gateformer.toml`](../../../../configs/models/Gateformer.toml).

## Differences

The official implementation (`models/Gateformer.py`, `layers/Embed.py`) was
inspected at the pinned revision (`298fd10db4af8ddab67e9d0044481279a2b11241`)
to resolve patch padding (`stride`-length end replication) and the two gate
equations exactly. The repository carries no `LICENSE` file at that
revision, so this card omits `code`/`revision`/`license` provenance fields;
`verification/models.toml` records the inspected files with pinned blob
URLs instead. The official fixed sinusoidal position table is used verbatim
(raw sin/cos amplitude); the reused `positional_encoding` component
standardizes that table (zero mean, unit-scaled variance) before use, a
minor magnitude difference that keeps the same relative-position content.
No source file was copied or adapted.

## Shared components

- [`flatten_forecast_head`](../_components/flatten_forecast_head/README.md)
- [`gated_fusion`](../_components/gated_fusion/README.md)
- [`positional_encoding`](../_components/positional_encoding/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `patch_len=16`, `stride=8`, `d_model=128`, `n_heads=8`, `e_layers=2`, `d_ff=256`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Paper

Gateformer combines cross-time and cross-variate attention through gated
representations rather than simple concatenation or a fixed architectural
order (paper Section 3, Figure 2): a global (inverted) embedding and a
patch-attention temporal-dependency embedding are blended by a learned gate
before cross-variate attention, and the pre- and post-cross-variate-attention
representations are blended by a second learned gate before the forecast
head.

## Source and verification

The official implementation (`models/Gateformer.py`, `layers/Embed.py`) was
inspected at the pinned revision (`298fd10db4af8ddab67e9d0044481279a2b11241`)
to resolve patch padding (`stride`-length end replication) and the two gate
equations exactly. The repository carries no `LICENSE` file at that
revision, so this card omits `code`/`revision`/`license` provenance fields;
`verification/models.toml` records the inspected files with pinned blob
URLs instead. The official fixed sinusoidal position table is used verbatim
(raw sin/cos amplitude); the reused `positional_encoding` component
standardizes that table (zero mean, unit-scaled variance) before use, a
minor magnitude difference that keeps the same relative-position content.
No source file was copied or adapted.

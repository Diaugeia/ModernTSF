---
name: "TimeExpert"
summary: "TimeExpert replaces vanilla self-attention in a channel-independent patch Transformer with Temporal Mix of Experts (TMOE): every key/value patch position becomes a candidate expert, each query differentiably routes to only its top-k most relevant experts, and one optional shared global expert preserves long-range context."
paper: "https://arxiv.org/abs/2509.23145"
paper_title: "TimeExpert: Boosting Long Time Series Forecasting with Temporal Mix of Experts"
venue: "arXiv"
year: 2025
code: "https://github.com/xwmaxwma/TimeExpert"
revision: "f53b5220f22767a91040aaa04679c2eb9b2eb9c5"
license: "unspecified (no LICENSE file at the pinned revision)"
---
# TimeExpert

TimeExpert replaces vanilla self-attention in a channel-independent patch Transformer with Temporal Mix of Experts (TMOE): every key/value patch position becomes a candidate expert, each query differentiably routes to only its top-k most relevant experts, and one optional shared global expert preserves long-range context.

<!-- model-card:canonical:start -->
## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 12, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2509.23145); title: TimeExpert: Boosting Long Time Series Forecasting with Temporal Mix of Experts; venue/year: arXiv / 2025
- [codebase](https://github.com/xwmaxwma/TimeExpert); revision: `f53b5220f22767a91040aaa04679c2eb9b2eb9c5`; license: `unspecified (no LICENSE file at the pinned revision)`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`configs/models/TimeExpert.toml`](../../../../configs/models/TimeExpert.toml).

## Differences

Clean-room implementation: confirmed. The TMOE routing/gathering/attention
data flow was re-derived from the pinned official file's class boundaries and
docstrings; no source lines were copied. The differentiable top-k local
expert attention itself was extracted into the cataloged, paper-neutral
`topk_expert_attention` component rather than kept model-local, since the
mechanism (score every key/value position, keep only the top-k per query,
optionally append one shared global expert) is not specific to any one
patch-embedding or head choice.

- The official code implements only "soft" or "none" routing-weight
  multiplication modes (`mul_weight`); this implementation always applies the
  top-k softmax weighting to the gathered key before the second attention
  softmax is taken over the gathered set, which is mathematically the
  official `mul_weight='none'` behavior composed with softmax attention
  (the router's softmax weights are not separately reapplied, matching
  `TMOE.forward` when `mul_weight != 'soft'`).
- The official multi-task `Model` supports `long_term_forecast`,
  `imputation`, `anomaly_detection`, and `classification`; only the
  forecasting path is implemented here, matching this catalog's scope.
- Official defaults use `d_model=512`/`n_heads=8`; the ModernTSF preset
  lowers `d_model` to 128 to match this catalog's other patch-Transformer
  presets, keeping `n_heads=8`, `patch_len=16`, `stride=8`, `topk=4`,
  `shared=False` unchanged.

## Shared components

- [`embed`](../_components/embed/README.md)
- [`flatten_forecast_head`](../_components/flatten_forecast_head/README.md)
- [`topk_expert_attention`](../_components/topk_expert_attention/README.md)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=12`. Default
model parameters are: `enc_in=7`, `d_model=128`, `n_heads=8`, `e_layers=2`, `patch_len=16`, `stride=8`, `dropout=0.1`, `topk=4`, `shared=False`
<!-- model-card:canonical:end -->

## Source and verification

Clean-room implementation: confirmed. The TMOE routing/gathering/attention
data flow was re-derived from the pinned official file's class boundaries and
docstrings; no source lines were copied. The differentiable top-k local
expert attention itself was extracted into the cataloged, paper-neutral
`topk_expert_attention` component rather than kept model-local, since the
mechanism (score every key/value position, keep only the top-k per query,
optionally append one shared global expert) is not specific to any one
patch-embedding or head choice.

- The official code implements only "soft" or "none" routing-weight
  multiplication modes (`mul_weight`); this implementation always applies the
  top-k softmax weighting to the gathered key before the second attention
  softmax is taken over the gathered set, which is mathematically the
  official `mul_weight='none'` behavior composed with softmax attention
  (the router's softmax weights are not separately reapplied, matching
  `TMOE.forward` when `mul_weight != 'soft'`).
- The official multi-task `Model` supports `long_term_forecast`,
  `imputation`, `anomaly_detection`, and `classification`; only the
  forecasting path is implemented here, matching this catalog's scope.
- Official defaults use `d_model=512`/`n_heads=8`; the ModernTSF preset
  lowers `d_model` to 128 to match this catalog's other patch-Transformer
  presets, keeping `n_heads=8`, `patch_len=16`, `stride=8`, `topk=4`,
  `shared=False` unchanged.

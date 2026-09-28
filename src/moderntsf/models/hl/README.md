---
name: "HL"
summary: "HL (Historical Last) is a naive spatiotemporal forecasting baseline that repeats the last observed value across every node and every step of the prediction horizon. It serves as a lower-bound reference in graph- and node-structured benchmarks, providing the simplest possible prediction without any learning."
paper: ""
paper_title: "Historical-last persistence baseline (no associated paper)"
venue: "N/A (classical baseline)"
year: null
code: "https://github.com/PoorOtterBob/CauAir"
revision: "73dae00ca6ad14abb15174a0a0286d500e868b94"
license: "NOASSERTION"
---
# HL

HL (Historical Last) is a naive spatiotemporal forecasting baseline that repeats the last observed value across every node and every step of the prediction horizon. It serves as a lower-bound reference in graph- and node-structured benchmarks, providing the simplest possible prediction without any learning.

<!-- model-card:canonical:start -->
## Method overview

HL (Historical Last) is a naive spatiotemporal forecasting baseline that repeats the last observed value across every node and every step of the prediction horizon.

## Core architecture

It serves as a lower-bound reference in graph- and node-structured benchmarks, providing the simplest possible prediction without any learning.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 12, nodes]`. The
declared output contract is a `[batch, 12, nodes]` point forecast. Adjacency and temporal/node covariates are supplied only when the model's executable contract requires them.

## Paper and code

- paper: not available; title: Historical-last persistence baseline (no associated paper); venue/year: N/A (classical baseline) / not available
- [codebase](https://github.com/PoorOtterBob/CauAir); revision: `73dae00ca6ad14abb15174a0a0286d500e868b94`; license: `NOASSERTION`

## Local implementation

ModernTSF implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/HL.toml`](../../../../configs/models/HL.toml).

## Differences

**Clean-room implementation: confirmed.** The exact persistence formula is
covered by deterministic equation and boundary tests. No reference source was
copied and there is no paper/checkpoint reference comparison claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=12` and `pred_len=12`. Default
model parameters are: `enc_in=8`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Historical Last (HL) is a naive persistence baseline for spatiotemporal forecasting. For each node in the graph, it copies the final observed value from the input window and repeats it identically across all future time steps in the prediction horizon. Despite its simplicity, the method provides a meaningful lower bound: any learned model should outperform HL, especially over longer horizons where temporal dynamics diverge from the last observation. Its forecast has no learned parameters.

## In ModernTSF
Default config: `configs/models/HL.toml`; model specification: `spec.py`;
clean-room implementation: `model.py`.

## Verification

**Clean-room implementation: confirmed.** The exact persistence formula is
covered by deterministic equation and boundary tests. No reference source was
copied and there is no paper/checkpoint reference comparison claim.

## Citation

HL is an in-repository persistence baseline and has no associated paper or
canonical BibTeX entry.

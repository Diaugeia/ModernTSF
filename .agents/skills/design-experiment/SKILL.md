---
name: design-experiment
description: Design a reproducible TSFLab experiment before execution. Use for selecting baselines, datasets, horizons, metrics, seeds, ablations, controls, budget, and stopping criteria; not for launching an already-defined config.
---

# Design an experiment

Turn a research question into validated run configs and a design table, without
spending compute. Design in the current Agent with native tools; no task renderer
or design command is required.

## Inputs

- The research question and any fixed constraints (datasets, models, budget).

## Steps

1. State the question and a falsifiable comparison first.
2. Resolve task mode, datasets and splits, representative horizons, primary and
   secondary metrics, strong baselines, controlled variables, seeds, resource
   budget, and failure or stopping criteria. Separate required comparisons from
   optional scale-up runs.
3. Encode shared settings by inheritance: a short run file that extends
   `base.toml`, one dataset preset, and one model preset (or `tsflab://configs/...`
   in a standalone project). Override only scientific variables under
   `experiment`, `task`, `training`, `model.params`, and `evaluation`; use
   `[sweep]` only for intended axes. The loader rejects unknown structural keys.
4. Keep ablations one factor at a time unless interactions are the question.
   Match preprocessing, training budget, evaluation strategy, and metric direction
   across models; disclose unavoidable capability differences.
5. Verify the resolved matrix, run count, and parameter variation through the
   config loader and preflight API (`uv run tsf inspect --config <run.toml>` is
   one optional adapter).

## Success

- Config paths plus a compact table: hypothesis, control, treatment, datasets,
  horizons, metrics, seeds, estimated runs, and acceptance criteria.

## Stop and hand off

- Do not launch costly runs unless requested; hand configs and budget to
  `run-experiment` once execution is authorized.
- Matching a published table belongs to `reproduce-paper-results`.

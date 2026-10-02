---
name: reproduce-paper-results
description: Reproduce and compare a forecasting paper's reported experiments in TSFLab by mapping its protocol to runnable configs and aligned metrics. Use for paper-result replication; not for implementing the model or designing an unrelated benchmark.
---

# Reproduce paper results

Rerun a paper's reported experiments under a traceable protocol map and compare
cell by cell. Success means the attempt is rerunnable and traceable, not that the
numbers match.

## Inputs

- An identified paper and a runnable, audited catalog model.
- The primary paper, supplement, authoritative source revision, and reported tables,
  all verified before spending compute.

## Steps

1. Route implementation doubts to `audit-model`; when the model is absent, render
   the `paper-to-model` task instead of implementing it here.
2. Map the protocol: dataset name and version, split boundaries, scaling, feature
   mode, lookback and horizons, covariates, loss, optimizer and schedule, batch
   size, epochs and stopping, seeds, checkpoint selection, metric formula and
   aggregation, baselines, and hardware-sensitive settings. Label every field
   `aligned`, `adapted`, `unknown`, or `blocked`; never fill gaps with defaults.
3. Encode aligned settings in dedicated inherited run configs, keep faithful
   replication separate from controlled adaptations, and inspect the matrix:

   ```bash
   uv run tsf inspect --config <paper-run.toml>
   ```

4. When authorized, execute through `run-experiment`; for multi-run work use the
   `paper-reproduction` task round and pass it with `tsf run --round`. Preserve raw
   outputs, resolved configs, environment facts, seeds, and failed runs.
5. Aggregate compatible cells with `analyze-results`, then report per cell: paper
   value, local value, absolute and relative difference, run count, uncertainty,
   and every protocol deviation. Missing or failed cells stay visible.

## Success

- A rerunnable, traceable comparison; call it reproduced only when the recorded
  protocol and results support that, otherwise partial or blocked.

## Stop and hand off

- Stop for a decision when missing data, licensing, ambiguous metrics, or
  infeasible compute would materially change the claim.

---
name: run-experiment
description: Design, preview, and run one or more TSFLab experiment or sweep configurations, in the repository or in a standalone project created with tsf init. Use for choosing baselines, seeds, and budgets, then training, evaluation, ablations, hyperparameter grids, concurrency, or GPU assignment; not for quick contract-only checks.
---

# Run experiments

Experiments module, step one: design, then execute validated configs and report
their artifacts. Run configs and `work_dirs/` records feed the result board that
AutoResearch reads. The Agent owns design and
interpretation; the library owns validation, execution, budgets, and recovery, so
do not reimplement them with ad hoc subprocesses or edited manifests. Python APIs
and the CLI below are equivalent.

## Inputs

- A research question or resolved run configs and the resource
  intent: jobs, GPUs, and an optional research round.
- For a standalone project, scaffold it with `uv run tsf init <dir>`; its run
  configs extend installed presets through `tsflab://configs/...` and write to
  the project's own `work_dirs/`.

## Steps

1. Without finished configs, design first per [design](references/design.md): a
   falsifiable comparison, fair baselines, seeds, and budget as inherited TOML.
   Then preview and launch:

   ```bash
   uv run tsf run configs/runs/<run>.toml --dry-run --json
   uv run tsf run configs/runs/<run>.toml [--round <round-id>] [--jobs N] [--gpus 0,1]
   ```

   Attach a round only when one was supplied or the task needs persistent
   research context. Use `--gpus` only after checking memory and device intent.
2. Before long runs, verify data (fetch missing published presets with
   `uv run tsf data download <preset>`), output location, seeds, horizons, evaluation
   strategy, and profiling. Keep sweeps in TOML.
3. For GIFT-Eval, read `uv run tsf data prepare --from gift --help`, fetch only the
   requested data, and preview `configs/runs/gift_eval_sweep.toml`; record dataset
   versions, horizons, model compatibility, budget, and the missing-series policy.
4. For budgets, GPU queueing, tracking, cancellation, or interrupted-run recovery,
   read [execution controls](references/execution.md); ordinary one-off runs do
   not need them.

## Chain

- Module: Experiments.
- Reads: dataset card (L1) and profile, model card (L1) and preset, the result board for baselines.
- Produces: resolved configs, run records `work_dirs/<dataset>/<model>/records/<run_id>.json`, optional round events.
- Hands off to: `analyze-results` (board), `submit-results` (records), `diagnose-experiment`, `run-autoresearch`.

## Success

- Every config reported as succeeded or failed, with its `work_dirs/` artifacts.

## Stop and hand off

- Never silently restart or overwrite a costly run.
- Failed, unstable, or suspect runs go to `diagnose-experiment`; complete,
  compatible outputs go to `analyze-results`. Execution alone does not establish
  a fair comparison.

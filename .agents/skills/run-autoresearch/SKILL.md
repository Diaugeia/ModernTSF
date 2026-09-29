---
name: run-autoresearch
description: Run a bounded iterative forecasting research loop from a falsifiable hypothesis through experiments and evidence-backed conclusions, composing cataloged components into candidate methods when authorized. Use when autonomous experiment iteration is authorized; not for a single predefined run or paper implementation.
---

# Run bounded autoresearch

Iterate hypothesis, experiment, and evidence inside hard budgets. The loop runs in
the current Agent; a rendered task, CLI call, or second Agent is not required.

## Inputs

- A research question, approved datasets, primary metric, hard run and time
  budgets, output location, and authorization to execute experiments.
- Separate authorization before writing model code; without it, iterate on configs
  and existing catalog methods only.

## Steps

1. Use `design-experiment` for a falsifiable comparison, fair baseline, seeds,
   controls, and stopping criteria before spending compute.
2. For durable budgets, reuse a supplied round or call
   `moderntsf.benchmark.infra.api.create_round` with the agreed limits and attach it to
   execution. `prepare_task` from a task template may supply defaults; it does not
   own the loop. Record only material decisions and evidence references.
3. With an iteration budget, call `claim_iteration(round_id, operation="<id>")`
   once before each iteration (CLI: `tsf research iteration <round-id> --operation
   <id>`); reusing an id is idempotent, and preparation or resume costs nothing.
4. Start with the cheapest experiment that can reject the hypothesis; run it with
   `run-experiment`, changing one factor per iteration unless an interaction is
   the question.
5. When authorized to build a candidate method, compose it from cataloged
   components (`uv run tsf component match <terms> --json`) in a new flat model
   through `add-model`, so it is verified and compared against the whole catalog
   under the same protocol.
6. Route failures to `diagnose-experiment` and compatible results to
   `analyze-results`. Preserve configs, seeds, environments, raw outputs, and
   failures; never overwrite a costly run.

## Success

- The hypothesis, run ledger, compatible metrics with uncertainty, failed or
  excluded cells, limitations, and a stop/continue recommendation.

## Stop and hand off

- Stop on budget exhaustion, repeated infrastructure failure, an invalid
  comparison, no measurable progress, or a conclusion that meets the acceptance
  criterion.
- External publication and additional task dispatch stay out of scope unless
  separately authorized.

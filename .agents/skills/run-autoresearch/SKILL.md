---
name: run-autoresearch
description: Run a bounded iterative forecasting research loop from a falsifiable hypothesis through experiments and evidence-backed conclusions. Use when autonomous experiment iteration is authorized; not for a single predefined run or paper implementation.
---

# Run bounded autoresearch

Require a research question, approved datasets, primary metric, hard run and time
budgets, output location, and authorization to execute experiments. Use
`design-experiment` to define a falsifiable comparison, fair baseline, seeds,
controls, and stopping criteria before spending compute.

Continue in the current Agent: formulate hypotheses, edit configs, interpret
evidence, and choose the next experiment with native reasoning and tools. A
rendered task, CLI call, or second Agent is not required to begin this work.

For durable cross-experiment budgets, reuse a supplied round or call
`moderntsf.benchmark.infra.api.create_round` with the agreed limits. Attach that round to
execution so limits are enforced across sweeps. An optional task template can
supply initial defaults through `prepare_task`; it does not own the research loop.
Record only material decisions and evidence references in the round. Native plans
and conversation need not be copied into another ledger.

Resolve and preflight every matrix through the library; CLI inspection is an
optional adapter. Start with the
cheapest experiment that can reject the hypothesis, then use `run-experiment`
with the round attached.
Preserve resolved configs, seeds, environments, raw outputs, and failures; never
overwrite a costly run. Route failures through `diagnose-experiment` and compatible
results through `analyze-results`.

When an iteration budget is declared, the Agent defines each research iteration
and calls `claim_iteration(round_id, operation="stable-iteration-id")` once before
its work. One iteration may contain several matrices; preparation and resume do
not consume iterations. Reusing the same operation ID is idempotent. The optional
CLI adapter is `tsf research iteration <round-id> --operation <id>`. Private
reasoning is not automatically metered.

Change one experimental factor per iteration unless an interaction is the stated
question. Continue only when the previous evidence justifies the next run. Stop on
budget exhaustion, repeated infrastructure failure, invalid comparison, no
measurable progress, or a conclusion strong enough for the declared acceptance
criterion.

Return the hypothesis, run ledger, compatible metrics with uncertainty, failed or
excluded cells, limitations, and a stop/continue recommendation. Repository model
code, external publication, and additional task dispatch remain out of scope unless
separately authorized.

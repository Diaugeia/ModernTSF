---
name: run-autoresearch
description: Run a bounded, profile-driven forecasting research loop: analyze the data, state hypotheses, run a baseline panel, search and recombine cataloged components, and conclude from evidence. Use when autonomous experiment iteration is authorized; not for a single predefined run or a paper implementation.
---

# Run bounded autoresearch

Iterate profile, hypothesis, experiment, and evidence inside hard budgets. The loop
runs in the current Agent; a rendered task, CLI call, or second Agent is not required.
Detail lives in the references; read each only at its step.

## Inputs

- A research question, approved datasets, primary metric, hard run, iteration, and
  time budgets, an output location, and authorization to execute experiments.
- Separate authorization before writing model code or registering a model; without
  it, search over configs and existing catalog methods only.

## Steps

1. Profile each approved dataset before choosing models:
   `uv run tsf dataset analyze <preset>` (or `--path FILE`). Read the profile and the
   fired catalog rules per [data-profile](references/data-profile.md).
2. Write 2-4 falsifiable hypotheses, each tied to a profile fact, with the metric
   change that would refute it. Use `design-experiment` for controls and seeds.
3. Open a round for durable budgets (`tsf research start --goal ... --max-runs N
   --max-iterations M`, or reuse a supplied one) and note each hypothesis
   (`tsf research note <id> --kind hypothesis`). Claim one iteration per phase with
   `tsf research iteration <id> --operation <phase>`; reuse of an id is free.
4. Retrieve candidates progressively: `uv run tsf catalog search <terms>` (L0 lines),
   `tsf <model|component|dataset> show <name>` (L1), `--depth 2|3` only when a
   decision needs it; see [retrieval-baselines](references/retrieval-baselines.md).
5. Evaluate a small baseline panel before any search (same split, horizon, metric,
   seeds); it sets the bar every later candidate must beat.
6. Search by changing one factor per iteration from the best baseline, starting with
   the cheapest run that can reject a hypothesis. Recombine cataloged pieces through
   the slot grid and prune from the profile per [recombination](references/recombination.md);
   `tsf component compose <spec.toml>` dry-runs a composition.
7. Confirm only finalists with the declared seeds, then compare against the baseline
   panel with spread, not a single best run.
8. When authorized and a recombined method wins, register it with `add-model` as a
   catalog model whose card records the composition and provenance, and verify it
   (`tsf verify model <Name>`, `tsf repo audit`).
9. Record decisions, runs, and conclusions as round events per
   [round-ledger](references/round-ledger.md); route failures to `diagnose-experiment`
   and compatible results to `analyze-results`. Never overwrite a costly run.

## Leakage guards

- Choose models, hyperparameters, and stopping by train and validation only; fix the
  selection rule before running. Test metrics confirm frozen candidates once.
- `shift.train_to_test` in a profile is a diagnostic; never act on it.

## Success

- Profile paths, hypotheses with verdicts, baseline panel, run ledger with seeds and
  uncertainty, excluded or failed cells, limitations, and a stop/continue recommendation.

## Stop and hand off

- Stop on budget exhaustion, repeated infrastructure failure, an invalid comparison,
  no measurable progress after two consecutive iterations, or an acceptance criterion met.
- External publication and task dispatch stay out of scope unless separately authorized.

---
name: curate-components
description: "Identify repeated or reusable operators across existing TSFLab models, prove semantic and runtime-contract equivalence, and extract them as cataloged components without changing behavior. Use for deliberate cross-model consolidation or preparing building blocks for automated research; not for reorganizing models into families or sharing code based on similar names."
---

# Curate shared components

Models module: turn duplicated or recombinable operators into cataloged components while keeping
every consumer's behavior bit-for-bit unchanged. Components are reusable
implementation units, never model categories; read the components section of
`.agents/STANDARDS.md` first.

## Inputs

- An explicit scope: named models, a family of candidates, or a repository pass.
- The current catalog, read progressively: `uv run tsf catalog search --kind component <terms>` (L0),
  `tsf catalog show <name>` (L1), `--depth 2` (full card), `--depth 3`
  (paths to open).

## Steps

1. Build a shortlist from model-local code (repeated classes, exact or normalized
   AST matches, recent additions). Lexical similarity only nominates candidates.
2. For each candidate compare equations, tensor axes and shapes, normalization and
   residual order, masking, initialization, state, dtype/device behavior, outputs,
   and error contracts. Decide `reuse-existing`, `extract-new`, or
   `keep-model-local`, and record why.
3. Prefer migrating to an existing component. Extract a new one when two or more
   consumers match exactly, or when the operator is a paper-neutral building block
   with a standalone contract worth recombining.
4. Before editing, capture reference fixtures from the unmodified models under a
   fixed seed and input: `state_dict()` keys and values, eval-mode outputs, and
   gradients. Keep parameter attribute names so checkpoints still load.
5. Define the smallest paper-neutral API; add its `ComponentSpec`, unit tests, and
   explicit imports; write the curated card (front matter and sections in the
   components section of `.agents/STANDARDS.md`) from the code, not from memory. Migrate consumers in reviewable groups without flags that
   hide material variants; keep attribution and explanatory comments.
6. Add a focused equivalence test against the fixtures, regenerate cards, and
   re-verify every touched model:

   ```bash
   uv run tsf repo cards
   uv run tsf model verify <Name...>
   uv run tsf repo check --contracts strict --models <Name...>
   uv run tsf model audit --components
   uv run tsf repo check --audit
   ```

## Chain

- Module: Models.
- Reads: component cards (L0-L2) and model-local code.
- Produces: component cards with interface, invariants, consumers; equivalence tests.
- Hands off to: `add-model` and `run-autoresearch` (recombination reads component interfaces), `audit`.

## Success

- Each new component has a curated card that passes `tsf model audit --components`, unit tests, and declared consumers.
- Every migrated consumer passes the equivalence test (keys, outputs, gradients).
- No peer-model imports; tests that pin the component count are updated.

## Stop and hand off

- If equivalence cannot be shown, leave the variant local and note the reason in
  its card; a no-change pass is valid.
- Report the scope inspected, decisions, consumers, and next candidates so a later
  pass can resume. This skill does not schedule background monitoring.

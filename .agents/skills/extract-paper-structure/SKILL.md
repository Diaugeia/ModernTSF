---
name: extract-paper-structure
description: Extract a forecasting paper's implementable architecture, equations, tensor contracts, official-code clarifications, training objective, component decisions, and ambiguities. Use before local implementation or structure audit; not for broad literature discovery.
---

# Extract paper structure

Produce an implementation map precise enough that another worker can implement or
audit the model without guessing a defining operation.

## Inputs

- The primary paper and supplement.
- The official repository, when one exists: pin a revision and record its license.
- The current component catalog (`uv run tsf component list`, one L0 line each).

## Steps

1. Keep paper facts, official-code clarifications, and local design choices in
   separate fields. Do not copy source text or code into the map.
2. Record, in execution order:
   - task, inputs, covariates, tensor axes, output, and probabilistic semantics;
   - preprocessing, normalization, decomposition, embeddings, main blocks, head,
     and inverse transforms;
   - defining equations with section or equation references;
   - loss, auxiliary objectives, initialization, defaults, train/eval differences;
   - shape invariants, sequence constraints, marks or adjacency contracts, edge cases;
   - official-code clarifications with revision and source path;
   - unspecified details that would materially change fidelity.
3. Give every implementable operation a component decision:
   `reuse-existing` (component and matched contract), `extract-new` (the
   standalone contract it would expose and its expected consumers), or
   `model-local` (the reason it is paper-specific). Check candidates with:

   ```bash
   uv run tsf component search <operation-and-contract-terms>   # L0 lines
   uv run tsf component show <candidate>                        # L1; --depth 2 only if needed
   ```

   Compare mathematics, axes, normalization, masking, residual order,
   initialization, state, and outputs; a similar name is not a match.
4. Mark paper-neutral operators (transforms, attention variants, routers,
   normalizations) as `extract-new` so they become reusable building blocks, per
   the components section of `.agents/STANDARDS.md`.

## Success

- A map covering every defining operation, each with a component decision.
- Facts and inferences are labeled separately; the official revision is pinned.

## Stop and hand off

- Stop and surface the ambiguity when choosing silently would change the method.
- Hand the map to `implement-model` (or `integrate-foundation-model` for a released
  pretrained runtime), or to `audit-model` for a structure audit.

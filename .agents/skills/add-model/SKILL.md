---
name: add-model
description: Turn one forecasting paper into an admitted flat-catalog model, from structure extraction and component decisions through local implementation, model card, spec, preset, verification manifest, and tests. Use for adding or replacing one locally implemented model; not for paper discovery, foundation checkpoints, or placeholder entries.
---

# Add a model

Models module, step two: create one flat catalog entry and admit it atomically once
it is implemented and verified. Models and methods are peers; no family directories.
Its cards (tagline, tags, composition, Key ideas) are the model context AutoResearch
retrieves.

## Inputs

- Public name, lowercase module slug, parameters, input needs, output type.
- Paper identity and, when official code exists, its URL, revision, and license
  together (a missing license is recorded, not guessed).

## Steps

1. Extract the paper structure and give every defining operation a component
   decision (`reuse-existing`, `extract-new`, `model-local`) per
   [references/paper-structure.md](references/paper-structure.md). Stop if an
   ambiguity would change the method.
2. Scaffold an unregistered workspace from the resolved facts; pass `--task-mode`
   explicitly when it is not ordinary `time_series`:

   ```bash
   uv run tsf model scaffold --name MyModel --paper-title "..." --paper-url <url> \
     --venue <venue> --year <year> --components revin \
     --params "enc_in:int,hidden:int=128" [--task-mode spatiotemporal]
   ```

3. Before implementing any block, `uv run tsf catalog search --kind component <terms>` (L0), open a
   candidate with `tsf catalog show <name>` (L1), and read `--depth 2|3` only to
   settle a doubt. Retrieval is a shortlist: verify shapes, axes, normalization,
   masking, residual order, initialization, state, and outputs before reusing.
4. Implement locally per [references/implementation.md](references/implementation.md);
   replace every scaffold placeholder and never import another named model package.
5. Fill the retrieval layer first (rules in STANDARDS "Progressive disclosure"):
   `tagline`, `tags`, the six-slot `composition`, `## Key ideas`; then inputs/outputs,
   links, local implementation, differences, components, and constraints. Put
   provenance in README front matter; keep `spec.py` to construction and runtime facts.
6. Add a preset (and a smoke preset when training is cheap), plus focused
   paper/equation and reference checks in `verification/models.toml` and `tests/`.
7. Admit atomically; admission registers the model, runs unified verification and
   all audits, and rolls back on failure:

   ```bash
   uv run tsf model add --name MyModel
   uv run tsf repo cards        # if components or dataset cards changed
   uv run tsf repo check --audit
   ```

## Chain

- Module: Models.
- Reads: component cards (L0 search, L1 interface) and dataset cards when inputs matter.
- Produces: model card (tagline, tags, six-slot composition, Key ideas), spec, preset `configs/models/<Name>.toml`, evidence under `verification/evidence/`.
- Hands off to: `run-experiment` (preset), `run-autoresearch` (catalog search and compositions), `audit`.

## Success

- One indexed card and runtime spec, a preset, passing evidence, truthful source
  and artifact facts, and every reused component declared in `spec.py` and card.
- Tests that pin catalog counts are updated to the new totals.

## Stop and hand off

- Generated placeholder code is never a catalog entry; stop if admission fails.
- Released pretrained runtimes use `integrate-foundation-model`; cross-model
  consolidation uses `curate-components`; do not widen into an unsolicited refactor.

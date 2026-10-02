---
name: add-model
description: Scaffold and admit a locally implemented forecasting model to the flat catalog after its paper structure and runtime contract are resolved. Use for the package, model card, spec, preset, verification manifest, and tests; not for paper discovery or placeholder catalog entries.
---

# Add a model

Create one flat catalog entry and admit it atomically once it is implemented and
verified. Models and methods are peers; do not create family directories.

## Inputs

- Public name, lowercase module slug, parameters, input needs, output type.
- Paper identity and, when official code exists, its URL, revision, and license
  together (a missing license is recorded, not guessed).
- A component decision for every defining operation (`extract-paper-structure`).

## Steps

1. Scaffold an unregistered workspace from the resolved facts:

   ```bash
   uv run tsf model scaffold --name MyModel --paper-title "..." --paper-url <url> \
     --venue <venue> --year <year> --components revin \
     --params "enc_in:int,hidden:int=128" [--task-mode spatiotemporal]
   ```

   Pass `--task-mode` explicitly when it is not ordinary `time_series`.
2. Before implementing any block, run `uv run tsf component search <terms>` per
   operation (L0 lines), open a candidate with `tsf component show <name>` (L1:
   interface and constraints), and read `--depth 2` or the source (`--depth 3`
   lists paths) only to settle a doubt. Retrieval is a shortlist: verify shapes,
   axes, normalization, masking, residual order, initialization, state, and outputs
   before reusing.
3. Implement with `implement-model`; replace every scaffold placeholder and never
   import another named model package.
4. Keep `spec.py` to factory, strict schema, config path, capabilities, declared
   components, and runtime contract; put descriptive and provenance facts in the
   README front matter. Complete the card's method, structure, inputs/outputs,
   links, local implementation, differences, components, and constraints.
5. Add a preset (and a smoke preset when training is cheap), plus focused
   paper/equation and reference checks in `verification/models.toml` and `tests/`.
6. Admit atomically; admission registers the model, runs unified verification and
   all audits, and rolls back on failure:

   ```bash
   uv run tsf model add --name MyModel
   uv run tsf repo cards        # if components or dataset cards changed
   uv run tsf repo audit
   ```

## Success

- One indexed card and runtime spec, a preset, passing evidence, truthful source
  and artifact facts, and every reused component declared in `spec.py` and card.
- Tests that pin catalog counts are updated to the new totals.

## Stop and hand off

- Generated placeholder code is never a catalog entry; stop if admission fails.
- Use `integrate-foundation-model` for a released pretrained runtime.
- Do not expand a single-model addition into an unsolicited refactor; use
  `curate-components` for cross-model consolidation.

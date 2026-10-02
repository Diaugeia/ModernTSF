---
name: audit-repository
description: Audit TSFLab for Agent-first assets, catalog drift, documentation consistency, generated cards, model construction, and forward contracts. Use before release or after structural changes; not for a single model-only check.
---

# Audit the repository

Confirm that code, catalogs, cards, evidence, and Agent assets agree across the
whole repository.

## Inputs

- The working tree to audit and, when known, the affected models.

## Steps

```bash
uv run tsf repo cards            # regenerate cards and indexes first, then inspect the diff
uv run tsf repo audit
uv run tsf verify stale
uv run tsf repo doctor --strict
uv run pytest -q
```

Run affected smoke configs as well. Check canonical Agent assets and links, flat
models, shared components, README front matter against runtime specs and presets,
and public `tsf` instructions. Every model needs local code, a readable card, a
manifest entry, and current evidence, with no classification fields,
undocumented model, persisted blocker, or failed verification. The strict doctor
covers forward execution, finite gradients, batch size one, and exact state-dict
and output round trips.

## Success

- All audit sections PASS, zero stale evidence, and a clean `repo cards` diff;
  otherwise failures reported by layer: assets, metadata, source facts,
  paper/reference checks, construction, contracts, smoke, formatting, tests.

## Stop and hand off

- Report failures; single-model repairs go to `audit-model`, cross-model
  consolidation to `curate-components`.

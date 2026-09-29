---
name: expand-model-catalog
description: Discover and integrate a bounded number of new time-series forecasting papers into the ModernTSF catalog. Use for an authorized search-to-model expansion run; not for literature monitoring that must leave the repository unchanged.
---

# Expand the model catalog

Run search, implementation, and admission end to end for a fixed number of papers.

## Inputs

- Authorization to modify the repository and an explicit candidate and model budget.

## Steps

1. Build the queue with `discover-papers` (arXiv and Hugging Face Papers,
   normalized identities, deduplicated against `uv run tsf model list --json`).
2. For each retained candidate, verify the primary paper, forecasting task,
   required inputs, official source, revision, and license (recorded as missing
   when absent), then select at most the authorized number.
3. Per paper: `extract-paper-structure` → `implement-model` (or
   `integrate-foundation-model` for a released pretrained runtime) → `add-model`.
   Build the `reuse-existing` / `extract-new` / `model-local` map before code, and
   extract paper-neutral operators as components.
4. Papers are independent: run them in parallel workers only on disjoint model
   directories, and merge shared aggregate files (catalog registry, verification
   manifest, component catalog, pinned counts) once at the end.
5. Each model passes focused tests, unified verification, strict runtime, model
   audit, component audit, and repository audit.

## Success

- Every admitted model verified with a truthful card; skipped candidates listed
  with reasons. No candidate clearing the gate is a valid no-change outcome.

## Stop and hand off

- Stop when the budget is exhausted or paper, data, or runtime ambiguity would
  require inventing a claim. Never use search relevance or a shape-only test as
  implementation evidence.
- Do not publish, push, open issues, or dispatch more tasks unless separately
  authorized.

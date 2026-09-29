---
name: discover-papers
description: Discover, deduplicate, rank, and optionally dispatch review tasks for new time-series forecasting papers. Use for arXiv or Hugging Face paper scans, recurring literature monitoring, and candidate-model intake; not for claiming or completing a paper reproduction.
---

# Discover forecasting papers

Find new forecasting methods, deduplicate them against the catalog, and return a
ranked, source-linked queue. Search relevance is never permission or evidence to
add an implementation. Read [references/intake.md](references/intake.md) before
scanning or dispatching.

## Inputs

- A time window and optional focus (task, architecture, data regime).
- The deduplication baseline: `uv run tsf model list --json` (paper URLs, titles,
  public names from the model cards).

## Steps

1. Search both arXiv and Hugging Face Papers across the query lattice in the
   reference rather than one broad phrase; prefer source metadata and primary
   paper or project pages over snippets.
2. Normalize arXiv identifiers and titles, collapse cross-source duplicates, and
   reject papers that do not forecast future time-series values (evaluation-only,
   explanation, or out-of-scope tasks such as irregular series).
3. Rank by task relevance, novelty against the flat catalog, authoritative code
   availability, recency, and implementability. A missing official license does
   not disqualify a paper; record it so the implementation is an independent rewrite.
4. Write one brief per retained paper, separating verified facts from inference.

## Success

- A ranked queue with primary URLs, deduplication results, and code facts; an
  empty queue with the scan recorded is a successful monitoring result.

## Stop and hand off

- Dispatch only when the user or recurring prompt explicitly asks: at most three
  independent tasks per run, one paper each, carrying the brief, URLs,
  deduplication result, deliverable, and the instruction to use `add-model` after
  paper, source, and runtime inputs are resolved. Dispatched tasks never merge,
  publish, or modify external systems.
- Implementation belongs to the downstream `paper-to-model` task or
  `expand-model-catalog`; search results never establish a local implementation.

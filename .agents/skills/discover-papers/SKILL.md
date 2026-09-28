---
name: discover-papers
description: Discover, deduplicate, rank, and optionally dispatch review tasks for new time-series forecasting papers. Use for arXiv or Hugging Face paper scans, recurring literature monitoring, and candidate-model intake; not for claiming or completing a paper reproduction.
---

# Discover forecasting papers

This skill is a research-intake harness. It finds candidates and creates bounded,
reviewable work items; it does not treat search relevance as permission or evidence
to add an implementation.

Read [references/intake.md](references/intake.md) before scanning or dispatching.

## Scan

1. Run `uv run tsf model list --json` and use model-card paper URLs, titles, and
   public names as the deduplication baseline.
2. Search both arXiv and Hugging Face Papers. Cover the query lattice in the
   reference instead of relying on one broad phrase. Prefer source metadata and
   primary paper/project pages over search snippets.
3. Normalize arXiv identifiers and titles, collapse cross-source duplicates, and
   reject papers that do not actually forecast future time-series values.
4. Rank candidates by task relevance, novelty relative to the flat catalog,
   authoritative code availability, license clarity, recency, and implementability.
5. Return a concise candidate brief for each retained paper. Clearly distinguish
   facts from inference; discovery does not claim implementation or verification.

## Dispatch

Dispatch only when the user or the recurring-task prompt explicitly requests it.
Create at most three independent tasks in one run, one paper per task. Include the
candidate brief, primary URLs, deduplication result, expected deliverable, and the
instruction to use `add-model` only after paper, source, license, and runtime inputs
are resolved. Do not ask a dispatched task to merge, publish, or modify external
systems.

Without dispatch authorization, report the ranked queue in the current task. If no
candidate clears the threshold, report that the scan completed with no dispatch;
unchanged state is a successful monitoring result.

## Integration boundary

The downstream task owns implementation. It must preserve the flat
`src/moderntsf/models/<lowercase_module_slug>/` layout, use shared components only when
semantics match, and pass the repository's provenance and contract gates. Search
results alone never establish a local implementation or verification result.

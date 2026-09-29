---
name: report-defect
description: Reproduce, diagnose, and draft a ModernTSF defect report. Use for bugs, regressions, incorrect model behavior, or repository failures; do not publish an issue or pull request without explicit authorization.
---

# Report a defect

Turn an observed failure into a minimal, evidence-backed report.

## Inputs

- The environment, exact public command, config, expected and observed behavior,
  traceback, and artifacts.

## Steps

1. Reproduce narrowly with the smallest config, then run the relevant checks:

   ```bash
   uv run tsf model show <Name>
   uv run tsf verify model <Name>
   uv run tsf smoke --model <Name>   # when the model has a smoke_config
   uv run tsf repo audit
   ```

2. Classify the failure: environment, configuration, dataset, catalog/spec,
   construction, forward, training, evaluation, or documentation, and identify
   the cause when possible.

## Success

- A draft with a minimal reproduction, evidence, classification, and suspected cause.

## Stop and hand off

- Do not implement a fix for a diagnosis-only request, and never publish an issue
  or pull request without explicit authorization.

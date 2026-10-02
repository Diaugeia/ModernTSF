---
name: smoke-models
description: Run fast end-to-end smoke checks for one or more TSFLab models. Use after implementation changes or when validating training and output shape quickly; not for exhaustive repository or paper audits.
---

# Smoke-test models

Exercise construction, a short training path, evaluation, and the declared output
shape on tiny data.

## Inputs

- The narrowest scope: one model, one config, or the whole catalog.

## Steps

```bash
uv run tsf smoke --model DLinear
uv run tsf smoke --config configs/runs/smoke_dlinear.toml
uv run tsf smoke --all --jobs 8
```

A model is smoke-testable only when `tsf model show <Name>` reports a
`smoke_config`; for the others, rely on `tsf verify model <Name>` and
`tsf repo doctor --strict --models <Name>`. Add focused cases for material
optional objectives or output types.

## Success

- Every selected config reports PASS; the final diagnostic of each failure is kept.

## Stop and hand off

- Smoke checks are not verification evidence. Failures go to
  `diagnose-experiment`; paper-level checks to `audit-model`.

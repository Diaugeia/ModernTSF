---
name: audit-model
description: Audit one existing ModernTSF model against its card, paper, official-code facts, local implementation, component decisions, and unified verification evidence. Use for a focused review; batch ownership belongs in a task harness.
---

# Audit a model

```bash
uv run tsf model show <Name>
uv run tsf model audit <Name>
```

Treat README front matter as canonical descriptive metadata. Check the card,
`spec.py`, preset, implementation, cited paper, codebase, license, and revision.
Compare equations, shapes, defaults, objective, preprocessing, initialization, and
output semantics. Confirm the implementation is local, the official revision was
inspected when available, external source was not copied or imported, and every
defining operation has a justified component decision.

For a model with runtime artifacts, confirm every required asset is pinned and
explicitly fetched, an artifact-aware factory receives only verified local paths,
offline absence fails before construction, and the card does not claim checkpoint
behavior when verification covers only random initialization.

For a declared inference-only foundation runtime, confirm it uses the official
loader through `src/moderntsf/models/_foundation/`, performs no implicit download, skips
training, and records unsupported training/gradient checks as `not-applicable`.
Do not require an official pretrained network to be rewritten locally.

Run `uv run tsf verify model <Name>` and inspect the generated evidence, then run
`uv run tsf repo doctor --strict --models <Name>`. If `model show` reports a
non-null `smoke_config`, also run `uv run tsf smoke --model <Name>`. Check finite
outputs, active gradients, state-dict round trip, CPU, batch and sequence bounds,
and declared marks or adjacency inputs. Confirm `reference_comparison` is executed
for official code or `not-applicable` only when none exists. Report failures instead
of persisting status or blockers in metadata.

When a task assigns several models, apply this complete audit independently to each
named model. Let the task own partitioning, parallel handoffs, shared-file writes,
and the final repository-wide gate.

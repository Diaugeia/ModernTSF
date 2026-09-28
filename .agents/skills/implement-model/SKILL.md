---
name: implement-model
description: Implement or replace one ModernTSF forecasting model as local code after checking its paper and pinned official implementation when available. Use after structure extraction and component matching; not for paper discovery, scaffolding alone, or experiment reproduction.
---

# Implement a model

Require a completed paper-structure map, resolved input/output contract, and an
explicit component decision for every defining operation. Inspect the primary
paper and supplement. When official code exists, pin its revision and inspect it
to resolve paper omissions such as tensor order, padding, initialization, defaults,
and train/eval behavior. Record its license. Never copy, rename, mechanically
rewrite, import, or depend on external model source.

If the requested outcome is the released pretrained foundation model rather than
a locally trained paper architecture, stop this workflow and use
`integrate-foundation-model`; the official runtime is deliberately not rewritten.

1. Design local modules from the extracted equations and verified implementation
   details. Mark each operation `reuse-existing`, `extract-new`, or `model-local`.
2. Reuse `src/moderntsf/models/_components/` only after proving mathematical and runtime
   equivalence. Keep paper-specific or semantically different blocks local.
3. Implement inside the flat `src/moderntsf/models/<slug>/` package. Preserve useful paper
   formulas and explanatory comments, but not source-derived code or comments.
   Keep the canonical four-input `forward` signature exact; model-specific
   operations belong in explicitly named methods rather than extra public inputs.
4. Keep `spec.py` limited to construction, parameter schema, config, capabilities,
   declared components, and runtime contract. Reject unknown parameters instead
   of accepting `**kwargs` or hidden aliases.
5. Complete the model card with paper facts, official code facts, local mapping,
   reused components, model-local blocks, differences, limits, and verification.
6. Add focused equation/structure checks. If official code exists, add a bounded
   `reference_comparison`; otherwise record that check as `not-applicable`.
7. For pretrained weights or tokenizers, declare checksum-pinned `ModelArtifact`
   entries in `spec.py`. Never download implicitly; inspect or fetch them with
   `tsf model artifacts`, and document whether the local path is optional or required.
   Add an explicit `artifact_factory(cfg, params, paths)` that constructs the local
   architecture from verified local paths; do not read an unpinned cache path or
   make the ordinary factory download assets.

Run:

```bash
uv run tsf model show <Name>
uv run tsf verify model <Name>  # existing model; new entries verify during model add
uv run tsf model audit <Name>
uv run tsf repo doctor --strict --models <Name>
uv run tsf component audit
uv run tsf repo audit
```

Success requires unified evidence covering every verification check, a readable
card, no peer-model implementation import, and no unresolved defining operation.
Stop rather than add a placeholder when the paper, inputs, or defining behavior is
too ambiguous to implement truthfully.

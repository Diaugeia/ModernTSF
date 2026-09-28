---
name: integrate-foundation-model
description: Integrate a released pretrained time-series foundation model through its official package and local checkpoint path. Use for zero-shot or inference-only foundation runtimes; not for ordinary paper architecture implementations.
---

# Integrate a foundation model

Use the official package, loader, configuration, and released weights. Do not
reimplement the pretrained network, copy its source, convert checkpoints without
a demonstrated need, or bundle weights in ModernTSF.

1. Confirm the primary paper, official repository, package version, checkpoint
   identifier, pinned revision, license, input semantics, outputs, and supported
   horizons. Treat an architecture-only local model as a different claim.
2. Check whether the official package can coexist with the main environment. If
   dependencies conflict, keep it in a compatible provider environment and use
   the same `src/moderntsf/models/_foundation/` boundary; do not relax core dependencies.
3. Reuse `FoundationModel` and the closest official runtime adapter. Load only
   from an explicit local path with offline mode enabled. Never trigger a network
   request during model construction, verification, or an experiment.
4. Add one ordinary flat `src/moderntsf/models/<slug>/` entry. Its factory receives verified
   local artifacts, declares `inference-only`, and exposes the canonical four-input
   interface. Do not add a provider registry or a foundation category.
5. Document official behavior, checkpoint facts, cache preparation, preprocessing,
   channel treatment, limitations, and every adapter transformation in the model
   card. Keep upper-layer methods as their own flat entries and compose them only
   through explicit experiment configuration.
6. Verify offline failure, official reference outputs, tensor axes, quantiles,
   finite values, CPU behavior where supported, batch/sequence boundaries, and
   state loading. Training and gradient checks are `not-applicable` only because
   the declared runtime is inference-only, not because verification was skipped.

Stop if the license, official loader, checkpoint identity, or input/output
semantics cannot be established. Adding the interface alone does not admit a
provider as a catalog model.

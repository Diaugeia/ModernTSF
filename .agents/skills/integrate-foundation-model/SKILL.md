---
name: integrate-foundation-model
description: "Integrate a released pretrained time-series foundation model through its official package and a pinned local checkpoint. Use for zero-shot or inference-only foundation runtimes; not for ordinary paper architecture implementations."
---

# Integrate a foundation model

Models module, variant of `add-model`: expose an official pretrained runtime as one flat, inference-only catalog entry.
Do not reimplement the network, copy its source, convert checkpoints without a
demonstrated need, or bundle weights in TSFLab.

## Inputs

- Primary paper, official repository, package version, checkpoint identifier,
  pinned revision, license, input semantics, outputs, and supported horizons.
- A local checkpoint path, or a checksum-pinned `hf://` artifact to fetch explicitly.

## Steps

1. Confirm identity and license; an architecture-only local model is a different
   claim and belongs to `add-model`.
2. Check whether the official package coexists with the main environment. If
   dependencies conflict, use a compatible provider environment behind the same
   `src/tsflab/models/_foundation/` boundary; never relax core dependencies.
3. Reuse `FoundationModel` and the closest official runtime adapter. Load only
   from an explicit local path with offline mode on; no network request during
   construction, verification, or an experiment.
4. Add one ordinary `src/tsflab/models/<slug>/` entry whose factory receives
   verified local artifacts, declares `inference-only`, and exposes the canonical
   four-input interface. Do not add a provider registry or a foundation category.
5. Document official behavior, checkpoint facts, cache preparation,
   preprocessing, channel treatment, limitations, and every adapter transformation
   in the card. Compose upper-layer methods only through experiment configuration.
6. Verify offline failure, official reference outputs, tensor axes, quantiles,
   finite values, CPU behavior where supported, batch/sequence bounds, and state
   loading; training/gradient checks are `not-applicable` because the runtime is
   inference-only, not because they were skipped.

```bash
uv run tsf model artifacts <Name>
uv run tsf model verify <Name>        # offline, CPU, small tensors: no training
uv run tsf repo check --contracts strict --models <Name>
```

## Chain

- Module: Models.
- Reads: official package facts, component cards (L0) for shared interfaces.
- Produces: an inference-only card, spec, pinned artifacts, preset, evidence (same layout as `add-model`).
- Hands off to: `run-experiment` (zero-shot runs), `run-autoresearch`, `audit`.

## Success

- One flat entry with pinned artifacts, offline construction, and evidence that
  matches the official reference outputs.

## Stop and hand off

- Stop if the license, official loader, checkpoint identity, or input/output
  semantics cannot be established; an adapter alone is not a catalog model.
- Admit the entry with `tsf model add --name <Name>` exactly as `add-model` step 7 does (unified verification, audits, rollback).

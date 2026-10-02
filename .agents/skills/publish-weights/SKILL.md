---
name: publish-weights
description: Package a completed TSFLab run's best checkpoint as a checksummed safetensors bundle and publish or load it through pinned hf:// URIs on the Hugging Face Hub. Use for sharing or reloading trained weights; not for leaderboard result submissions or pretrained foundation checkpoints.
---

# Publish weights

Release module: turn a finished run into a reproducible weights bundle at
`<dataset>/<model>/<run_id>/` (manifest, `model.safetensors`, run record, card),
addressed by a URI pinned to an immutable revision.

## Inputs

- A completed run id (or its `record.json`) with a checkpoint under
  `work_dirs/<dataset>/<model>/checkpoints/<run_id>/`.
- A target model repository (default `Diaugeia/TSFLab-Weights`) and write
  credentials (`HF_TOKEN` or a local Hub login), plus explicit authorization to
  publish. Requires the `hub` extra.

## Steps

1. Build the bundle locally and inspect it:

   ```bash
   uv run tsf hub pack <run_id>            # -> work_dirs/_bundles/<run_id>/
   ```

   Check `manifest.json`: model, dataset, horizon, seed, metrics, framework
   version, commit, and the SHA-256 of every file.
2. Publish only that run; repositories are created private unless `--public`:

   ```bash
   uv run tsf hub push <run_id> --repo <owner>/<repo> [--create] [--public]
   ```

   Record the printed `hf://<owner>/<repo>@<revision>/<path>` URI; the revision is
   the new commit, so the URI always resolves to the same bytes.
3. Reference the URI where the weights matter, for example a real-time forecast's
   `weights_uri`, and verify it round-trips:

   ```bash
   uv run tsf hub list --repo <owner>/<repo> --dataset <dataset>
   uv run tsf hub pull hf://<owner>/<repo>@<revision>/<dataset>/<model>/<run_id>
   ```

## Chain

- Module: Release.
- Reads: the run record and its checkpoint.
- Produces: a pinned `hf://<owner>/<repo>@<revision>/...` URI with a checksummed manifest.
- Hands off to: `forecast-realtime-round` (`weights_uri`), `run-experiment` (reload), `submit-results`.

## Success

- A pinned URI whose pull verifies every tensor against the manifest checksums.

## Stop and hand off

- Never publish without explicit authorization or push runs that were not
  requested; stop on checksum mismatch.
- Result bundles for the leaderboard belong to `submit-results`; released
  pretrained runtimes to `integrate-foundation-model`.

---
name: setup-environment
description: Install, repair, or verify the TSFLab Python environment and PyTorch backend, including the optional hub and realtime extras. Use for first-time setup, dependency failures, CUDA detection problems, or hardware changes.
---

# Set up the environment

Experiments module, step zero: produce a working, reproducible environment and report what was selected.

## Inputs

- The machine's hardware and the features needed (core, `hub` for weights
  bundles, `realtime` for live data sources, trackers only when requested).

## Steps

```bash
bash scripts/detect_hardware.sh
UV_TORCH_BACKEND=auto uv sync --python 3.12            # add --extra hub --extra realtime as needed
uv run tsf env audit --json
uv run tsf --help
uv run tsf repo audit
```

Use an explicit backend (`cpu`, `cu121`, ...) only when auto-detection is wrong or
reproducibility requires it. For a specific experiment, check readiness with
`uv run tsf env audit --config <run.toml> --json`; the audit reports facts and never
changes the environment.

## Chain

- Module: Experiments.
- Reads: hardware facts and `tsf env audit` output.
- Produces: a reported environment (versions, backend, extras) recorded with runs.
- Hands off to: `run-experiment`, `diagnose-experiment`.

## Success

- Python and torch versions, the selected backend, accelerator visibility, the
  installed extras, and any lockfile change are reported.

## Stop and hand off

- Do not change dependency pins to mask a driver mismatch; report it.
- Environment-caused run failures go to `diagnose-experiment`.

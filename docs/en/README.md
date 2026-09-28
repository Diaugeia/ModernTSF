# ModernTSF documentation

← [Project README](../../README.md)

Human documentation is intentionally small. Command syntax and defaults are
available from `uv run tsf --help` and each subcommand's `--help`; model,
component, and dataset details live in their cards.

- [workflows.md](workflows.md): model interfaces, adding models, offline official
  foundation runtimes, artifacts, components, datasets, verification, and experiments.
- [execution.md](execution.md): optional environment audits, tracking, budgets, GPU scheduling, recovery, and independently usable Python modules.
- [hub.md](hub.md): standalone projects (`tsf init`), `hf://` addressing, and publishing weights bundles.
- [models.md](models.md): generated flat catalog linking every model card.

Quick start:

```bash
UV_TORCH_BACKEND=auto uv sync --python 3.12
uv run tsf model list
uv run tsf dataset list
uv run tsf inspect --config configs/runs/run_single_data.toml
uv run tsf run configs/runs/run_single_data.toml
```

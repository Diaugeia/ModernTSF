# TSFLab documentation

← [Project README](../../README.md)

Human documentation is intentionally small. Command syntax and defaults are
available from `uv run tsf --help` and each subcommand's `--help`; model,
component, and dataset details live in their cards.

- [workflows.md](workflows.md): model interfaces, adding models, offline official
  foundation runtimes, artifacts, components, reading the catalog by depth, datasets
  (101 presets, one TSFLab protocol each), experiments, AutoResearch, and verification.
- [execution.md](execution.md): optional environment audits, tracking, budgets, GPU scheduling, recovery, and independently usable Python modules.
- [hub.md](hub.md): standalone projects (`tsf init`), `hf://` addressing, and publishing weights bundles.
- [realtime.md](realtime.md): rolling real-time tracks, weekly rounds, forecast submissions, scoring, and replay.
- [models.md](models.md): generated flat catalog linking every model card.

Quick start:

```bash
UV_TORCH_BACKEND=auto uv sync --python 3.12
uv run tsf catalog
uv run tsf catalog list --kind dataset
uv run tsf run configs/runs/run_single_data.toml --dry-run
uv run tsf run configs/runs/run_single_data.toml
```

## Commands

Eleven commands, one per module (`tsf <command> --help`):

| Command | Purpose |
| --- | --- |
| `tsf catalog` | overview, `search`, `list`, `show <name> [--kind] [--depth]` for models, components, datasets |
| `tsf data` | `add`, `prepare [--from traffic\|ultratraffic\|gift]`, `inspect`, `analyze`, `plot`, `download`, `publish`, `audit` |
| `tsf model` | `scaffold`, `add`, `artifacts`, `verify`, `compose`, `audit [--components]` |
| `tsf run` | run configs; `--smoke`, `--dry-run`, `--backend local\|queue\|slurm` |
| `tsf env` | environment audit; `storage` and `usage` subcommands |
| `tsf result` | `aggregate`, `rank`, `plot`, `report`, `predictions`, `board`, `submit`, `leaderboard`, `hub` |
| `tsf realtime` | rolling tracks: `update [--bootstrap] [--push]`, `open`, `forecast`, `score` |
| `tsf research` | research rounds: `start`, `list`, `show`, `note`, `status`, `iteration` |
| `tsf repo` | `check [--audit] [--contracts LEVEL]`, `cards`, `schema` |
| `tsf agent` | `task`, `interface`, `modules`, `sync` |
| `tsf init` | scaffold a project for chosen modules |

Install by module with extras: `data`, `models`, `experiments`, `hub`, `realtime`,
`autoresearch`, `all`.

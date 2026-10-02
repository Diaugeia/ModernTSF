# Retrieve candidates and run the baseline panel

## Progressive retrieval

Read the shortest card level first and escalate only for survivors:

1. Names and one-line summaries: `uv run tsf catalog list --kind model --json`,
   `uv run tsf catalog list --kind component`, plus lexical shortlists
   (`uv run tsf catalog search --kind model <terms>`, `uv run tsf catalog search --kind component <terms> --json`).
   Retrieve at the shortest card level the CLI offers. Start from the fired rules'
   `models` and `components`, then widen with these searches.
2. L1 for the shortlist (about 8 candidates): `uv run tsf catalog show <name>` gives a
   model's tagline, composition, key ideas, and runtime facts (`config`, `smoke_config`,
   `task_modes`, `output_type`), or a component's interface, public symbols, and consumers.
3. L2 (`--depth 2`, the full card) only for at most 3 finalists; read the method,
   structure, inputs/outputs, and differences. `--depth 3` lists the files to open.

Drop a candidate as soon as a shallow level excludes it: wrong task mode (time_series vs
spatiotemporal vs covariate), a required input the data lacks, an inference-only or
artifact-gated model, or capacity far above the train length. Note each exclusion and its
reason as a round `decision` event.

## Baseline panel

Run before any search, on the same split, horizon, lookback, metric, and seeds:

- `NLinear` or `RLinear` (normalized linear anchor) and `DLinear` (decomposition anchor).
- `PatchTST` (channel-independent deep anchor).
- `iTransformer` (channel-mixing anchor) when `train.cross_channel` is applicable.
- `ExpSmoothingTS` or `ARIMATS` when forecastability is low, as a classical floor.
- Plus the single best model the fired rules name, if it is not already in the panel.

Use presets in `configs/models/<Name>.toml`; a run file extends `configs/base.toml`, one
`configs/datasets/<preset>.toml`, and one model preset, and overrides only scientific
variables. Preview the matrix with `uv run tsf run <run.toml> --dry-run` before any
execution; when authorized, the run machine (GPU host or CI, not the local machine)
executes `uv run tsf run <run.toml> --round <round-id>`. Prefer one horizon and one
seed for the panel; the panel costs at most a third of the run budget.

The panel result fixes the bar: the best baseline per dataset (by validation loss, then
confirmed on test) and the spread across baselines. If a simple baseline is already within
seed noise of the best, say so; deep models then need a large margin to be worth keeping.

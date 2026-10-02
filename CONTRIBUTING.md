# Contributing to TSFLab

Thanks for helping grow the benchmark! This guide covers the common contributions:
proposing or adding a model, submitting results, forecasting real-time rounds,
adding a dataset, and reporting issues.

## Branching & releases

- **`dev` is the integration branch — open your PRs against `dev`, not `main`.**
  New models, datasets, bug fixes, and features all land on `dev` first.
- **During the 1.0 release-candidate phase, `dev` is the RC line.** It carries
  `1.0.0rcN` builds; each RC is tagged `v1.0.0rcN` on `dev`, and `main` moves
  only when a candidate is promoted to `v1.0.0`. Fixes for an RC land on `dev`
  through pull requests like any other change.
- **`main` is release-only and versioned.** It is protected (a PR is required to
  merge, the `schema-check` status check must pass, force-pushes and deletion are
  blocked). `main` normally advances only by promoting `dev` → `main`, and
  **every `main` update bumps the version (`pyproject.toml` + `CHANGELOG.md`) and
  ships a tagged GitHub Release** (`vX.Y.Z`, [semver](https://semver.org/)):
  bug-fix-only promotions bump the patch, new models/features bump the minor.
- A maintainer may push an urgent hot-fix straight to `main` (admin bypass), but
  it still must carry the version bump + release.
- Branch names follow the conventional scopes used in history: `feat/…`,
  `fix/…`, `docs/…`, `chore/…`.

## Setup

```bash
# The PyTorch build (CPU vs CUDA) is chosen at install time via UV_TORCH_BACKEND.
# Let uv auto-detect, or pin explicitly (cpu / cu121 / cu124 / ...).
UV_TORCH_BACKEND=auto uv sync --python 3.12
bash scripts/detect_hardware.sh   # reports the recommended backend
```

Do **not** add a hardcoded `+cuXXX` torch pin to `pyproject.toml` — it breaks
CPU/macOS installs. The backend is selected via `UV_TORCH_BACKEND`.

## Reporting issues

Open an issue from the templates — **Submit a new model**, **Report a bug**, or
**Ask for a feature**. The forms require the context we need (repro config,
environment, official-source license, …); issues without it may be closed.

## Proposing a method (no code required)

Open a **Submit a new model** issue with the paper link. When a maintainer adds
the `intake-approved` label, the `paper-intake` workflow has a coding agent
implement the paper as a catalog model, re-runs the verification battery
independently, and opens a pull request for review. The same workflow scans the
literature weekly.

## Submitting results and real-time forecasts

- **Benchmark results:** add a `submission.json` under `apps/web/submissions/`
  as described in [`apps/web/SUBMITTING.md`](apps/web/SUBMITTING.md). CI
  validates it against the TSF-Core contract.
- **Real-time rounds:** add `forecasts/<YourModel>.json` to an open round under
  `apps/web/submissions/realtime/<track>/rounds/<round_id>/` before its
  deadline; CI rejects pull requests last updated after the deadline. See
  [`docs/en/realtime.md`](docs/en/realtime.md).

## Adding a model

See the [model workflow](docs/en/workflows.md#add-a-model-or-method). In short:

1. Deduplicate and extract the paper; inspect pinned official code when available.
2. Match defining operations against `src/tsflab/models/_components/`.
3. Run `tsf model scaffold` with paper/source facts and component decisions.
4. Implement locally, complete the card, and declare focused manifest tests.
5. Run `tsf model add --name <Name>`; atomic admission performs verification and
   rolls catalog registration back if a gate fails.

## Adding a dataset

See the [data workflow](docs/en/workflows.md#data).

## Verifying

Every model needs unified evidence and strict runtime checks:

```bash
uv run tsf verify model <Name>
uv run tsf repo doctor --strict --models <Name>
uv run tsf repo audit
```

The final repository gate requires CPU construction, forward, backward, boundaries,
active gradients, finite outputs, and state-dict round trips; do not waive a failed
contract with documentation.

## Licensing

The project is MIT (see [`LICENSE`](LICENSE)). Models are maintained as local
implementations after checking papers and official code; external model source is
not vendored. Record dependency notices in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

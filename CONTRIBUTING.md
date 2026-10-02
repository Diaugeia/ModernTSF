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
  merge, the `ci` checks (`gate`, `wheel`, `web`) must pass, force-pushes and deletion are
  blocked). `main` normally advances only by promoting `dev` → `main`, and
  **every `main` update bumps the version (`pyproject.toml`) and
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

Open an issue from one of two templates: **Add a paper** or **Report a problem
or ask a question**. An agent classifies it and replies. A reproducible problem
gets a fix through the same check, review, and merge path as any other change;
include the command, config, traceback, and environment so it can be reproduced.

## Proposing a method (no code required)

Open an **Add a paper** issue with the paper link; the agent chooses the model
name. `agent.yml` triages it and posts the decision. An accepted paper is
implemented as a catalog model; the workflow independently runs
`tsf repo check`, a second read-only agent reviews the change, and the pull
request is merged when both pass (`INTAKE_AUTOMERGE=false` leaves merging to
maintainers). The same workflow scans the literature weekly.

## CI and the agent maintainer

> **Current state:** CI runs the static checks and the test suite on every push
> and pull request; model verification (`ci.yml` → *Run workflow* → `verify`),
> deployment, release, and the agent and weekly workflows are started manually
> until their secrets are configured. The rest of this section describes the
> full setup they return to.

Three workflows, organized by module:

- `ci.yml`: on every push and pull request, `tsf repo check` (the single
  definition of mergeable: schema, agent assets, cards, verification staleness,
  repository audit, web submissions, pytest, and the affected-model smoke run on
  pull requests), the wheel check, and the web build. A push to the deploy branch
  (`DEPLOY_BRANCH`, default `main`) deploys the site; a `v*` tag builds, creates
  the GitHub release, and publishes to PyPI.
- `agent.yml`: weekly paper discovery, issue handling, and contributor pull
  request review. Every merge needs `tsf repo check` to pass and an agent
  approval; issue and PR text is treated as untrusted data. Pull requests that
  touch workflows, agent assets, dependencies, scripts, or the check itself are
  left for a maintainer. Set `AGENT_PAUSED=true` to stop every agent job.
- `weekly.yml`: the real-time cycle (merged after `tsf repo check`) and the
  pinned-data link check.

Run `uv run tsf repo check --scope changed` before opening a pull request.

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
2. Fix the retrieval-layer facts first, because readers find a model through them:
   a `tagline` (at most 120 characters), `tags` (including one architecture family),
   and the six-slot `composition`
   (`normalization|decomposition|temporal|channel|head|loss`). Match the defining
   operations against existing components with `tsf catalog search --kind component` and
   `tsf catalog show`.
3. Run `tsf model scaffold` with the paper/source facts and component decisions.
   It creates an unregistered workspace.
4. Implement locally, complete the card (including `## Key ideas`), and declare
   focused tests in `verification/models.toml`.
5. Run `tsf model add --name <Name>`. Admission registers the model, regenerates
   the cards, runs verification and the audits, and rolls the registration back if
   any gate fails.

## Adding a dataset

See the [data workflow](docs/en/workflows.md#data). A dataset has a preset, a
card, and exactly one TSFLab protocol (split, scaling, lookbacks, horizons) in the
card's `protocol` field; the protocol used in the literature, when different, goes
in `literature_protocol`. Dataset files live under `dataset/` and are never
committed. Smoke and synthetic inputs under `configs/fixtures/` are test fixtures,
not datasets. Check a new card with `tsf data audit` and profile the data with
`tsf data analyze <preset>`.

## Verifying

Every model needs unified evidence and strict runtime checks:

```bash
uv run tsf model verify <Name>
uv run tsf repo check --contracts strict --models <Name>
uv run tsf repo check --audit
```

The final repository gate requires CPU construction, forward, backward, boundaries,
active gradients, finite outputs, and state-dict round trips; do not waive a failed
contract with documentation.

## Documentation

Human documentation (this file, `README.md`, `docs/en/`, and the resource cards)
describes public behavior and CLI workflows. Cards and generated tables are
projections of code: change the model, spec, config, or schema first, then run
`uv run tsf repo cards`. `uv run tsf repo check --audit` checks that the documentation
and cards agree with the code.

## Releases

There is no changelog file. Each tagged release is a GitHub Release whose notes
are generated from the merged pull requests, so write clear pull-request titles.

## Licensing

The project is MIT (see [`LICENSE`](LICENSE)). Models are maintained as local
implementations after checking papers and official code; external model source is
not vendored. Record dependency notices in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

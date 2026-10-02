# Repository standards
Read only the section relevant to the change; `AGENTS.md` holds the always-on rules.

## Models

Every model or method is a peer under `src/tsflab/models/<lowercase_module_slug>/`
with no architecture categories. Each entry owns `model.py`, a checked `README.md`
model card, and a `spec.py` limited to its factory, parameter schema, config path,
and runtime contract. The catalog index joins registered specs to card front matter;
registration is the admission boundary, and configs are presets, not registrations.
A releasable entry must import, validate parameters, construct, return finite
correctly shaped output, and pass unified verification.

## Components

Reusable building blocks live in `src/tsflab/models/_components/`; they never
classify models. Every new model maps its defining operations to
`reuse-existing`, `extract-new`, or `model-local` first. Reuse a component only
when mathematics, shapes, normalization, masking, residual order, initialization,
and outputs match; similar names are not evidence. Extract a new one when two or
more consumers match exactly, or when it is a paper-neutral building block with a
standalone contract (transform, attention variant, router, normalization) that
automated research can recombine; paper-specific glue stays local.

Each extraction needs unit tests and affected-model contract tests; extraction from an
existing model preserves state-dict keys, outputs, and gradients against prior fixtures.
Material variants stay local and named. Models never import peer-model code; shared code
moves into a component with a card. A component card is curated: flat front matter (`name`, `kind`, `module`, `summary`,
`category`, `input`, `output`, `origin`, `origin_models`, `tags`) plus the sections
Purpose, Origin and granularity, Interface (every public symbol, shapes with axis
names, state), Invariants and equivalence evidence (cite existing tests/fixtures),
Variants and options, When to use and when not to use, Related components. Only the
`component-card:generated` block (API, import, consumers) is rendered; the audit
enforces both.

## Progressive disclosure
Start at `tsf catalog` (what exists), search L0 lines (`tsf catalog search`: name,
kind, tagline, tags), open L1 (`tsf catalog show`: front matter, key ideas or interface,
constraints), escalate to L2 card or L3 files only when a decision needs it. A model card
adds `tagline` (<=120 chars), `tags` (with one architecture family), `composition`
(six slots `normalization|decomposition|temporal|channel|head|loss=component:/local:/loss:/none`,
naming only imported components), and a `## Key ideas` section before the generated block.

## Model cards and sources
Each model `README.md` is the descriptive source of truth: a flat fact header, a body
mapping operations to local code, and differences from paper and official code. Required front matter is `name`, `summary`, `paper`, `paper_title`, `venue`, and
`year`. When official code exists, add `code`, `revision`, and `license` together;
otherwise omit all three. Do not add nested mappings, persisted verification status, empty
code fields, or invented source facts. Ordinary paper architectures are local code; inspect official code at a pinned revision
to resolve omissions without copying it. A released pretrained foundation model is the
exception: use its official package and checkpoint behind `src/tsflab/models/_foundation/`,
load offline from an explicit local path, and declare it inference-only. A shape-only smoke
test is not verification; status is computed from evidence, never written into the card.

## Verification
One route: `verification/models.toml` declares each model's paper checks, source
comparison when applicable, and runtime profile; `verification/evidence/<Model>.json`
records the result; `verification/index.json` is regenerated. Checks cover paper
structure, equations, construction, forward, backward, finite outputs, active gradients,
state-dict round trip, CPU, batch/sequence bounds, input contract, and reference
comparison (official code, else `not-applicable`). Use `tsf model verify <Name...>|--all|--stale|--index`.

## Data, experiments, and model artifacts
Dataset bytes live only in `dataset/`, loaders in `src/tsflab/data/`, cards in
`catalog/datasets/`. Task modes are executable contracts checked at config loading.
Experiments are resolved TOML and immutable evidence under `work_dirs/`; execution policy never
changes scientific settings. Large weights are `ModelArtifact` facts in `spec.py`, pinned by
revision and SHA-256, never bundled or fetched implicitly (`tsf model artifacts`).

## Dataset cards
Curated facts plus one `dataset-card:canonical` generated block (loader, files, parameters).
Flat `key: <JSON>` front matter: `summary`, `domain`, `tags`, `source`, `source_url`, `citation`,
`citation_url`, `license`, `redistribution`, `frequency`, `time_span`, `length`, `channels`,
`channel_kind`, `target`, `missing_values`, `protocol`, `seq_lens`, `pred_lens`, `split`,
`stats_basis` (`measured|source-reported|mixed`), `related`, optional `realtime_track`. Body:
Overview, Provenance and license, Structure and statistics, Standard protocol and known
pitfalls, generated block, Related datasets. Measure statistics from local files, else cite a
primary source; write `unknown` for unverifiable licenses; never invent facts.

## Documentation ownership
Human docs (root, `docs/`, cards) explain public behavior without Agent paths; Agent
procedures live only under `.agents/`. Schemas, front matter, specs, configs, and
tests are executable truth; generated indexes are projections: update code truth
first, then regenerate.

## Modules, skills, and tasks
Chain: Data -> Models -> Experiments -> Release; each module yields context AutoResearch
reads (cards, profiles, compositions, interfaces, result board, round ledger).

- Data: add-dataset, inspect-dataset.
- Models: discover-papers, add-model, integrate-foundation-model, curate-components (task `intake`).
- Experiments: setup-environment, run-experiment, diagnose-experiment, reproduce-paper-results, analyze-results (task `experiment`).
- Release: submit-results, forecast-realtime-round, publish-weights.
- AutoResearch: run-autoresearch (task `autoresearch`).
- Maintenance: audit, handle-contribution (tasks `maintenance`, `contribution`).

Skills live only at `.agents/skills/<skill-name>/SKILL.md` (80-line budget; detail in
`references/`), kebab-case `name`, discriminating `description`, and Inputs, Steps,
Success, Stop sections using public commands only. Tasks are `.agents/tasks/<name>.toml`
(`tsf agent task`). Check with `uv run python -m tsflab.tsf_core.agent_assets`.

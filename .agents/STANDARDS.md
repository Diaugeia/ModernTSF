# Repository standards

Read only the section relevant to the current change. `AGENTS.md` contains the
always-on rules; this file keeps detailed contracts out of the default context.

## Models

Every model or method is a peer under `src/moderntsf/models/<lowercase_module_slug>/`
with no architecture categories. Each entry owns `model.py`, a checked `README.md`
model card, and a `spec.py` limited to its factory, parameter schema, config path,
and runtime contract. The catalog index joins registered specs to card front matter;
registration is the admission boundary, and configs are presets, not registrations.
A releasable entry must import, validate parameters, construct, return finite
correctly shaped output, and pass the unified verification contract.

## Components

Reusable building blocks live in `src/moderntsf/models/_components/`; they never
classify models. Every new model maps its defining operations to
`reuse-existing`, `extract-new`, or `model-local` first. Reuse a component only
when mathematics, shapes, normalization, masking, residual order, initialization,
and outputs match; similar names are not evidence. Extract a new one when two or
more consumers match exactly, or when it is a paper-neutral building block with a
standalone contract (transform, attention variant, router, normalization) that
automated research can recombine; paper-specific glue stays local.

Each extraction needs unit tests and affected-model contract tests; extraction
from an existing model preserves state-dict keys, outputs, and gradients against
fixtures captured beforehand. Material variants stay local and named. Models never
import peer-model code; shared code moves into a component with a generated card.
Avoid catch-all utilities and flag-driven base classes that hide paper behavior.

## Model cards and sources
Each model `README.md` is the descriptive source of truth. Its front matter is a
flat, human-readable fact header. The body maps defining operations to local code
and states differences in preprocessing, architecture, objective, training, output,
and defaults.
Required front matter is `name`, `summary`, `paper`, `paper_title`, `venue`, and
`year`. When official code exists, add `code`, `revision`, and `license` together;
otherwise omit all three. Do not add nested mappings, persisted verification
status, empty code fields, or invented source facts.
Ordinary paper architectures are maintained as local code. Inspect authoritative
official code at a pinned revision when available to resolve paper omissions,
without copying it. A released pretrained foundation model is the narrow exception:
use its official package and unchanged checkpoint behind `src/moderntsf/models/_foundation/`,
load offline from an explicit local path, and declare the flat catalog entry
inference-only. Record source and license facts. A shape-only smoke test is not
verification, and verification status is computed from evidence rather than
written into the card.

## Verification
There is one route named `verification`. `verification/models.toml` declares each
model's paper checks, source comparison when applicable, and special runtime
profile. `verification/evidence/<Model>.json` records the complete result;
`verification/index.json` is regenerated and never hand-maintained.
Required checks cover paper structure, equations, construction, forward, backward,
finite outputs, active gradients, state-dict round trip, CPU, batch and sequence
boundaries, input contract, and reference comparison. Reference comparison uses
official code when available and is otherwise `not-applicable`; it is a check, not
a classification. Use `tsf verify model`, `stale`, `all --jobs`, and `index`.

## Data, experiments, and model artifacts
Dataset bytes live only in `dataset/`; loaders and schemas in `src/moderntsf/data/`; cards in
`catalog/datasets/`. Task modes are executable contracts checked during config loading.
Experiments are resolved TOML and immutable evidence under `work_dirs/`; execution policy never
changes scientific settings. Large weights and tokenizers are `ModelArtifact` facts in `spec.py`,
pinned by revision and SHA-256, never bundled or fetched implicitly (`tsf model artifacts`).

## Dataset cards
A card is curated facts plus one generated block between `<!-- dataset-card:canonical:start/end -->`;
`tsf repo cards` rewrites only that block and `name`, `kind`, `config`, `loader`, `alias`,
`task_modes`. Flat `key: <JSON>` front matter (levels 0/1) holds `summary` (one specific
sentence), `domain`, `tags`, `source`, `source_url`, `citation`, `citation_url`, `license`,
`redistribution` (`allowed|conditional|restricted|unknown`), `frequency`, `time_span`, `length`,
`channels`, `channel_kind`, `target`, `missing_values`, `protocol`, `seq_lens`, `pred_lens`,
`split`, `stats_basis` (`measured|source-reported|mixed`), `related`, optional `realtime_track`.
Body (level 2): Overview, Provenance and license, Structure and statistics, Standard protocol
and known pitfalls, generated block, Related datasets. Measure statistics from local files,
else cite a primary source as `source-reported`; write `unknown` for unverifiable licenses and
never invent facts. A loader family (GIFT-Eval) adds a family card. `tsf dataset audit` enforces it.

## Documentation ownership
Human documentation lives at the repository root, under `docs/`, and in resource
cards. It explains public behavior and CLI workflows without Agent paths, prompt
syntax, or internal command modules. Agent procedures live only under `.agents/`
and never duplicate human tutorials. Schemas, card front matter, specs, configs, and
tests are executable truth; generated indexes are projections. Update code truth
first, then regenerate or revise the English projection; do not hand-maintain
facts that a catalog can render.

## Skills
Skills live only at `.agents/skills/<skill-name>/SKILL.md`, with standard
kebab-case `name` and discriminating `description` frontmatter. Each skill owns
one recognizable outcome, expected inputs, preflight checks, execution path,
success criteria, artifacts, and stopping conditions. Use native Agent work and
public APIs or optional CLI adapters; omit harness-specific paths, assumptions, retired
aliases, internal script entry points, or copies of human-facing tutorials.
Changed skills must pass `uv run python -m moderntsf.tsf_core.agent_assets` and the standard
skill frontmatter validator. Test descriptions against positive, indirect,
incomplete, negative, and edge-case requests.

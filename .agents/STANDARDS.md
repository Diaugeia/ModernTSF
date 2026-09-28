# Repository standards

Read only the section relevant to the current change. `AGENTS.md` contains the
always-on rules; this file keeps detailed contracts out of the default context.

## Models

Every model or method is a peer under `src/moderntsf/models/<lowercase_module_slug>/`;
there are no architecture categories or separate method hierarchy. Each entry
owns `model.py`, a checked `README.md` model card, and a `spec.py` limited to its
factory, parameter schema, config path, and runtime contract.

The catalog index joins registered specs to model-card front matter; registration
is the admission boundary. Configs are runnable presets, not registrations. A
releasable entry must import, validate parameters, construct, return finite
correctly shaped output, and pass the unified verification contract.

## Components

Reusable building blocks live in `src/moderntsf/models/_components/`; they never classify
models. Extract code only when all consumers share mathematical behavior,
shapes, normalization, masking, residual order, initialization, and output
structure. Similar names are not evidence of equivalence.

Every new model maps its defining operations to `reuse-existing`, `extract-new`,
or `model-local` before implementation. Reuse a cataloged component whenever its
semantic and runtime contract matches. Only extract a new component when no
existing contract fits and multiple real consumers remain.

Every extraction needs focused unit tests and affected-model contract tests.
Material variants remain local and explicitly named. Named model packages must
not import implementation code from peer models; proven shared code moves into
a cataloged component with a generated README card. Avoid catch-all utility
modules and flag-driven base classes that conceal paper-specific behavior.

## Model cards and sources
Each model `README.md` is the descriptive source of truth. Its front matter is a
flat, human-readable fact header. The body maps defining operations to local code
and states differences in preprocessing, architecture,
objective, training, output, and defaults.

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
Local dataset bytes live only in `dataset/`; loaders and schemas live in
`src/moderntsf/data/`; readable preset cards live in `catalog/datasets/`. Dataset and model
task modes are executable contracts checked during config loading. Experiments are
resolved TOML and immutable evidence under `work_dirs/`; optional execution policy
never changes scientific settings. Recovery preserves identity and attempt history.
Large weights and tokenizers are `ModelArtifact` runtime facts in `spec.py`, pinned
by source revision and SHA-256. They are never bundled or downloaded implicitly.
Use `tsf model artifacts` to inspect or explicitly fetch them; required artifacts
must verify before construction. This capability does not classify a model.

## Documentation ownership
Human documentation lives at the repository root, under `docs/`, and in model
cards. It explains public behavior and public CLI workflows without Agent paths,
prompt syntax, or internal command modules. Agent procedures live only under
`.agents/` and must not duplicate human tutorials. Schemas, model-card front
matter, runtime specs, configs, and tests are executable truth; generated indexes
and tables are projections.

Update code truth first, then regenerate or revise the English human projection.
Do not hand-maintain facts that can be rendered from a catalog.

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

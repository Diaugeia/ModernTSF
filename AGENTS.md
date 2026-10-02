# TSFLab Agent Guide

This is the canonical, harness-neutral Agent entrypoint and the default for
Codex. `.agents/skills/` contains workflows, `.agents/tasks/` bounded task
templates, and `.agents/STANDARDS.md` on-demand contracts. Pi and DeepSeek
Harness use these Agent assets directly. Claude Code uses `CLAUDE.md` and
`.claude/skills` links to the same files. Never duplicate the instructions.

## Invariants
- Preserve the flat `src/tsflab/models/<model>/` layout. Do not classify models or
  methods into architecture-family directories.
- Put proven reuse and paper-neutral building blocks in `src/tsflab/models/_components/`;
  paper-specific operations remain model-local. Components never form a model hierarchy.
- Verify paper and official-code claims before describing an implementation as
  faithful. Pin the inspected revision and record every material difference.
- Treat each model `README.md` front matter as the canonical descriptive and
  provenance record. `spec.py` owns construction, parameter schema, config path,
  and runtime facts only.
- Implement ordinary paper architectures locally after checking the paper and
  pinned official code. Released pretrained foundation models use thin, offline
  official-runtime adapters; their source and weights are never copied.
- Use one verification route only: `verification/models.toml`, generated
  `verification/index.json`, and one evidence file per model.
- Agent owns reasoning and decisions; APIs own execution guarantees; CLI is optional.
- Use an optional research round for multi-step experimental memory and budgets;
  ordinary catalog, verification, and one-off run workflows remain stateless.
- Preserve user data and generated experiment outputs unless their removal is
  explicitly requested.
- External issues, pull requests, publication, and dispatch require explicit
  authorization.

## Information layers
- Human-facing material lives in `README.md`, `CONTRIBUTING.md`, English `docs/`,
  and resource cards. Keep it task-oriented and limited to public APIs.
- Agent-only procedures live in `.agents/`, never presented as product docs.
- Descriptive truth lives in cards; runtime truth in schemas, specs, configs, and
  tests. Read cards progressively: `tsf catalog`, `tsf catalog search`, `tsf <kind>
  show <name>`, and `--depth 2|3` only when a decision needs it.

## Work and verification
Use the matching Skill for repeatable work. Read the relevant section of
`.agents/STANDARDS.md` only for structural, provenance, or Skill changes. Set up
and verify with:
```bash
UV_TORCH_BACKEND=auto uv sync --python 3.12
uv run tsf repo audit
```
Run narrow checks, affected smoke checks, and `tsf repo doctor --strict --models
<Name...>`; run it unscoped before release. Code, contracts, cards, and evidence
must agree.

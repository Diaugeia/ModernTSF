# Changelog

All notable changes to ModernTSF are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semantic versioning.

## [1.0.0rc1] — 2026-09-28 (release candidate)

A fully automated, continuously updated forecasting platform: one installable
package, one repository, one address for published assets, and rolling
real-time evaluation.

### Added

- `apps/web/`: the ModernTSF Leaderboard (formerly TSEval) with its history,
  submission pipeline and `submissions/`; CI validates submissions on PRs.
- `moderntsf.hub` and `tsf hub {pack,push,list,pull}`: pinned `hf://` URIs,
  SHA-256 verified downloads, and safetensors weights bundles
  (`<dataset>/<model>/<run_id>/`) for Hugging Face model repositories.
- `ModelArtifact` accepts pinned `hf://` URIs.
- `tsf init` scaffolds a standalone project; run configs can extend installed
  catalog presets through `moderntsf://` paths.
- `hub` optional dependency group (`huggingface_hub`, `safetensors`).
- Rolling real-time evaluation (`moderntsf.realtime`, `tsf realtime`): versioned
  append-only panels mirrored to the Hub, weekly rounds opened before their
  targets, baseline and catalog-model forecasts, scoring with frozen
  normalization, Kendall-tau rank stability, and historical replay. Tracks:
  `stock_hs300` (AKShare log returns), `traffic_pems_sb` (PeMS District 8,
  bootstrapped from UltraTraffic_CL), `air_openaq_cn` (OpenAQ PM2.5).
- TSF-Core `RoundSpec`, `ForecastSubmission`, `RoundScore` contracts and one
  shared leaderboard implementation (`tsf_core.leaderboard`).
- Workflows: `realtime-weekly` (CPU cron), `paper-intake` (Codex literature scan
  and implementation with independent verification; community requests through
  the new-model issue form), real-time forecast checks in `web-validate`.
- `realtime` optional dependency group (`akshare`, `requests`, `pyarrow`).
- 20 new methods (178 → 198) added through the paper-to-model workflow:
  SDMixer, SEMixer, LSINet, Dualformer, DPWMixer, AWEMixer, TimeExpert,
  WDformer, TQNet, TimePro, Gateformer, CANet, ReFocus, FreqMoE, SWIFT,
  Sensorformer, VisiFold, Extralonger, ST-SSDL, RAGC.
- 23 new shared components (24 → 47), extracted from the new methods and from
  existing models so that methods can be recombined: e.g. `wavelet`,
  `mixer_block`, `gated_dilated_conv`, `adaptive_node_embedding_adjacency`,
  `topk_expert_router`, `last_value_center`, `freq_band_moe`,
  `periodic_query_bank`, `node_visibility`, `deviation_memory`. 26 existing
  models now consume shared components; equivalence tests pin their state_dict,
  outputs, and gradients to the pre-refactor behavior.
- UltraTraffic PeMS datasets (`ultratraffic_st` / `ultratraffic_ts`): hourly flow
  for Caltrans Districts 3, 4, 7, 8 (2003–2023) with continual-learning splits;
  9 presets (89 in total) and real-time tracks `traffic_pems_{ba,la,sac,sb}`.

### Changed

- **Breaking:** all runtime packages moved under `moderntsf`
  (`moderntsf.benchmark`, `.data`, `.models`, `.tsf_core`, `.assets`); the
  wheel no longer installs generic top-level `benchmark`/`data`/`models`
  packages. Sources now live in `src/moderntsf/`.
- `tsf submit` points contributors at `apps/web/submissions/` in this repository.
- Pytest collects only `tests/`; `experiments/` is a local-only workspace.
- The site pipeline validates with the TSF-Core pydantic contracts; its
  hand-written `contract.schema.json` and the duplicate aggregator are removed.
- `build_model_meta.py` reads the current model-card fields.

- Agent skills: 24 (new `forecast-realtime-round`, `publish-weights`); every
  skill now follows one structure (Inputs, Steps, Success, Stop and hand off),
  covers real-time rounds, hub weights, standalone projects, and UltraTraffic, and
  uses public commands only. New public commands `tsf repo cards` and
  `tsf dataset convert-ultratraffic` replace internal script entry points.
- Component policy: paper-neutral building blocks with a standalone contract may
  be extracted even with a single consumer, so automated research can recombine
  them; behavior-preserving extraction is checked against pre-change fixtures.

### Fixed

- `tsf model add` compared a set of declared components with a tuple of imported
  ones, so admitting any model that declares components always failed.
- Scaffold templates (`tsf model scaffold`, `tsf dataset add`) generated imports
  from the removed top-level packages.
- Interrupted stock fetches now resume from a per-symbol cache.

## [0.8.0] — 2026-09-04

Composable execution services that augment the current Agent while keeping
ordinary experiments lightweight.

### Added

- Optional environment audits, execution policies, run/sweep status, cancellation,
  atomic checkpoints, training-batch and fixed-window evaluation recovery, and
  resumable LatentTSF pretraining.
- Persistent priority queues with crash recovery, shared or exclusive GPU leases,
  explicit Slurm transport, storage inspection and checkpoint-cleanup previews.
- Run, time, GPU-hour and external token/USD budgets with durable reservations.
- Optional TensorBoard/W&B mirrors, local scalar evidence and prediction figures.
- Lazy Python infrastructure APIs, module discovery, scoped policy schemas and
  shared API/CLI result envelopes. Queue executors can be injected in-process or
  referenced by importable module path for background work.
- Offline official-runtime boundaries for pretrained foundation models.

### Changed

- Agent-native planning, diagnosis, interpretation and reporting remain in the
  current Agent; templates, persistent rounds and CLI adapters are optional.
- Research iteration boundaries are explicitly Agent-defined and idempotent.
- Preflight and execution resolve copies of caller-owned policies. Model-owned
  pretraining accepts stage injection without importing execution infrastructure.

### Fixed

- Conflicting or incomplete round-budget inputs, config inheritance cycles and
  concurrent research-state updates.
- Complete config snapshots, atomic/idempotent result writes, report subprocess
  failures, protocol isolation and missing-seed visibility.
- Affected-model CI path extraction, data/core triggers, runner lazy imports and
  platform-sensitive gradient assertions for softmax- and LayerNorm-invariant biases.

### Compatibility

- Existing basic `tsf run` commands remain supported. LatentTSF's advanced
  pretraining recovery uses `stage_runner` injection.
- Batch recovery requires replayable input with zero data-loader workers; rolling
  evaluation restarts its traversal. Stateful external adapters supply state hooks.
- GPU sharing is cooperative admission. External spending is reported by callers;
  machine-startup services belong to the host. Slurm requires a shared environment.

See [execution controls](docs/en/execution.md) for module APIs and operational details.

## [0.7.0] — 2026-09-02

A focused Agent-infrastructure release that adds lightweight research rounds,
strengthens verified artifact construction, and keeps installed Harness assets
small and directly readable.

### Added

- Added optional research rounds with bounded run budgets, append-only decisions,
  complete command logs, and explicit experiment association.
- Added `tsf agent task start` to prepare a task prompt and research round without
  dispatching an external Agent.
- Added explicit verified-artifact factories for model runtimes that consume
  local weights, tokenizers, or other pinned assets.

### Changed

- Consolidated English-first, harness-neutral instructions for Codex, Pi,
  DeepSeek, and Claude Code links.
- Reduced wheel assets to model cards and literal specifications instead of a
  duplicate copy of every installed model implementation.

### Fixed

- Clarified task preparation versus Agent dispatch and added a preimplementation
  gate to the paper-to-model Harness.
- Updated contribution forms and affected-smoke CI to the flat model layout,
  shared-component path, unified verification commands, and English-only docs.

## [0.6.0] — 2026-08-28

A repository-architecture release that standardizes every model as a local
ModernTSF implementation, makes verification a single first-class workflow, and
hardens model, dataset, experiment, component, and Agent extension contracts.

### Added

- Added one declarative verification manifest, one evidence document per model,
  and generated indexes for all 178 models, exposed through `tsf verify model`,
  `stale`, `all`, and `index`.
- Added atomic model admission, catalog drift checks, strict experiment records,
  and task harnesses for paper discovery, model integration, component curation,
  verification backlogs, experiments, autoresearch, and repository audits.
- Added readable cards and audit coverage for the flat model catalog, 24 shared
  model components, and 80 dataset presets.

### Changed

- Rewrote the previously ported model set as repository-owned implementations
  after checking papers and pinned official references. External repositories are
  provenance and verification references, not runtime implementation sources.
- Moved reusable forecasting components under `src/models/_components/` while
  keeping paper-specific operations local to each flat model package.
- Simplified model-card metadata, dataset locations, runtime entrypoints, and
  experiment configuration; unknown or drifting options now fail closed.
- Made `AGENTS.md`, `.agents/skills/`, and `.agents/tasks/` the canonical
  harness-neutral Agent interface used by Codex, Pi, DeepSeek, and Claude links.
- Reduced the wheel to required runtime, cards, verification, configuration, and
  Agent assets instead of shipping the complete development repository.

### Removed

- Removed upstream/rewrite model classifications, split parity/rewrite
  validation routes, historical batch verification scripts, approximation
  adapters, retired training compatibility hooks, and obsolete command aliases.
- Removed duplicate documentation and generated metadata that could drift from
  executable catalogs and schemas.

### Fixed

- Stabilized verification evidence so unchanged runs no longer rewrite files or
  invalidate stale checks through timestamp-only changes.

## [0.5.0] — 2026-08-27

An Agent-first repository release that makes all 178 models auditable, removes
the legacy approximation/adapter system, and exposes reusable model, dataset,
component, experiment, and catalog-expansion capabilities through one packaged
repository interface.

### Added

- Added canonical readable cards for all 178 models, 80 dataset presets, and 24
  reusable components. Dataset and component cards are generated from executable
  catalog/configuration truth and are checked by the repository audit.
- Added component and dataset discovery commands, including machine-readable
  list/show/search/match/audit workflows for Agents.
- Added 9 bounded task harnesses for paper monitoring, paper-to-model
  integration, catalog expansion, component curation, verification, experiments,
  autoresearch, reproduction, and final repository audit.
- Added a packaged read-only repository snapshot so installed wheels retain the
  canonical Agent skills, task harnesses, cards, configs, tests, and verification
  evidence.

### Changed

- Made `AGENTS.md` and `.agents/skills/` the canonical Agent-first instruction
  surfaces. `CLAUDE.md` and `.claude/skills` are links only.
- Replaced per-model `registry.py` and `schema.py` files with one flat
  `ModelSpec` catalog. Models and methods share the same namespace and are not
  partitioned by architecture family.
- Promoted reusable implementation code to cataloged `src/components/` modules
  and removed the approximation-adapter backend in favor of verified clean-room
  rewrites or pinned licensed upstream ports.
- Replaced legacy scripts and aliases with the grouped `tsf` CLI. No legacy
  command or per-model registry compatibility layer is retained.
- Consolidated the repository workflow into 23 focused Agent skills and 9
  bounded task harnesses, including paper discovery, catalog expansion,
  experiment execution, autoresearch, and final repository audit.

### Fixed

- Hardened executable contracts across all 178 models, including initialization,
  output-horizon, segmentation, finite-adjacency, and graph-threshold issues.

## [0.4.0] — 2026-07-01

A feature release: **176 → 178 models** plus an opt-in training-objective hook
system for models whose loss is not the plain prediction error.

### Added

- **Opt-in custom training-objective hooks in the trainer** (`#21`). Models may
  now, without touching the default path used by every existing model, declare:
  - `train_loss_override` — a model-owned scalar loss that *replaces* the
    configured criterion for training (validation/early-stopping still use the
    configured observation loss);
  - `requires_train_target` + `set_train_target(y_or_none)` — the raw future
    target is fed only for the training forward and cleared before
    validation/evaluation (no leakage);
  - `pretrain(train_loader, device)` — a one-shot model-owned pretraining stage
    run before optimizer construction, with its wall time folded into
    `train_time_sec` / `fit_time`.
  The existing additive `aux_loss` convention is unchanged. Documented in
  `docs/en/add-model.md` (+ ZH) and the `add-model` / `smoke` skills.
- **CRIB** (`#24`) — new point model: forecast directly from partially observed
  series (MTSF-M) via a TCN + unified-variate Transformer information-bottleneck
  latent, with consistency and KL regularizers (rides the `aux_loss` convention).
- **GlocalIB** (`#25`) — new point model: Glocal Information Bottleneck alignment
  regularizer (NeurIPS 2025), forecasting port aligning the clean-input
  embedding with an augmented-view embedding (rides the `aux_loss` convention).
- **Probabilistic forecasting** (`#20`) — an opt-in `output_type` axis
  (`"point"` / `"quantile"` / `"distribution"`), orthogonal to `task.mode`.
  Quantile models emit non-crossing quantiles via a shared monotone
  `QuantileHead`; distribution models emit `(loc, scale)`. Scored with
  `crps`/`wql`/`coverage_80`/`width_80` alongside the existing point metrics.
  Ships with `TiRex`, `QuantileDLinear`, `QuantilePatchTST`, `MQRNN`,
  `GaussianMLP`, the `quantile`/`nll_gaussian` losses, and the
  `probabilistic-forecasting` skill. See `docs/en/probabilistic-forecasting.md`.

### Changed

- **LatentTSF** (`#22`) reimplemented as a faithful two-stage model (ICML 2026):
  a frozen per-timestep MLP autoencoder pretrained via the `pretrain()` hook plus
  a DLinear backbone forecasting in latent space via `train_loss_override`. The
  module was renamed `src/models/latentsf` → `src/models/latenttsf`.
- **TimeAlign** (`#23`) reimplemented as a faithful distribution-aware glocal
  alignment forecaster (ICLR 2026), injecting its 3-term objective through the
  `requires_train_target` / `train_loss_override` hooks.

### Hardened

- The trainer's DataParallel fail-fast guard now covers all custom-objective
  hooks (`requires_train_target` / `set_train_target` / `train_loss_override`),
  not just `requires_train_target`, and warns instead of silently discarding a
  non-finite/non-scalar `train_loss_override`.

## [0.3.3] — 2026-06-17

A bug-fix release.

### Fixed

- **`inverse=true` no longer crashes in `MS` mode.**
  `ForecastingDataset.inverse_transform` broadcast-failed when the model emits a
  single target column (`MS` = multivariate-input, single-target) while the
  scaler was fit on all `C` channels (`non-broadcastable operand (N,1) vs
  (N,C)`). A reduced-channel output is now anchored on the last `k` channels'
  statistics (the target channel(s), which time-series datasets place last);
  full-width `M`-mode inversion is byte-identical. Added
  `configs/runs/smoke_dlinear_msinv.toml` as a regression smoke.

## [0.3.0] — 2026-05-31

An agent-automation release: **99 → 115 models**, a single unified `tsf` tooling
entry point with concurrent scaffold / verify / run, one-shot Markdown reports,
and end-to-end smoke coverage for every model.

### Added

- **16 new CauAir air-quality models (99 → 115)** ported from
  [PoorOtterBob/CauAir](https://github.com/PoorOtterBob). These are the
  non-duplicate subset of the original 31-model port — the ~15 models that main
  had already ported independently from BasicTS (GWNet, STGCN, STID, DCRNN,
  DGCRN, STGODE, CrossGNN, D2STGNN, SOFTS, STNorm, STAEformer, TimeXer, UMixer,
  DSFormer, Pathformer) were skipped to avoid duplicate implementations.
  - Graph spatiotemporal: ASTGCN, GCLSTM, DeepAir, STTN, GAGNN, PM25_GNN,
    AirFormer, DSTAGNN, PCDCNet, AirPhyNet, AirDualODE
  - Non-graph baselines / forecasters: HL, LSTM, RPMixer, MGSFformer, CATS
  - Adapters wire each model to the `(cfg, params)` factory + `params["adj_mx"]`
    injection convention; a shared `src/models/_external/graph_utils.py` provides
    adjacency helpers.
- **Full smoke coverage (115/115).** Every model now has a
  `configs/runs/smoke_*.toml`; the 31 classic base models that previously lacked
  one were added, so `tsf smoke --all` is a concurrent end-to-end CI gate over
  the whole model zoo.

### Tooling — unified `tsf` Agent entry point

- **`tsf`** — one standard-library entry point (no extra deps, run via
  `uv`, concurrent where it helps) for every tool:
  - `new-model` / `new-dataset` — one-command scaffold (package + schema +
    registry + config + smoke config + `MODEL_NAME_MAP` entry). `--graph` emits a
    spatiotemporal variant.
  - `smoke` — concurrent end-to-end PASS/FAIL verification (`--all` / `--model` /
    `--config`, `--jobs N`); non-zero exit on any failure.
  - `run` — concurrent multi-config runner (`--jobs`, `--gpus`).
  - `aggregate-plot` — aggregate + bubble chart in one shot.
  - `report` — generate a shareable Markdown report (leaderboard + bubble chart +
    results table) for a dataset.
  - forwards verbatim to every other `tool/*.py`.
- New `smoke` and `report` Agent Skills; `add-model` / `add-dataset` / `sweep`
  skills now lead with the `tsf` scaffold.
- Retired the `run_multi_configs.sh` and `aggregate_and_plot.sh` shell scripts
  (replaced by `tsf run` / `tsf aggregate-plot`); only `detect_hardware.sh` remains.

### Dependencies

- Added `torchdiffeq` (ODE solvers for AirPhyNet / AirDualODE).
- Declared `reformer-pytorch` (Reformer) and `pywavelets` (STWave) in
  `pyproject.toml` — these were previously imported but never declared, so a
  clean `uv sync` would prune them and break those models.

### Docs

- Repo-wide consistency/accuracy pass across README (en+zh), `docs/` (en+zh),
  CLAUDE.md, and the skills; model count updated to 115.
- Slimmed the README badge row to `Python 3.12+ · uv · PyTorch 2.6 · Time Series
  Forecasting · Models 100+ · License MIT`.

**Full diff:** https://github.com/Diaugeia/ModernTSF/compare/v0.2.0...v0.3.0

## [0.2.0] — 2026-05-30

A large expansion release: **31 → 99 models**, three forecasting data settings,
a graph-adjacency pipeline, new metrics/losses/training tricks, and a full
docs + Agent-Skills refresh. The project is now **forecasting-only** at every
reachable code path.

### Added

- **68 new models (31 → 99)** ported from BasicTS (Apache-2.0), Time-Series-Library
  (MIT), and TFB, plus the PoorOtterBob set. Highlights:
  - Transformers: TimeXer, Crossformer, Informer, Transformer, Reformer, Pyraformer,
    ETSformer, NSTransformer, MultiPatchFormer, PAttn, CARD, Fredformer, DUET,
    Pathformer, DSFormer, DTAF, TimePerceiver
  - MLP/Patch: WPMixer, MTSMixer, UMixer, NHiTS, NBeats, HDMixer, SRSNet
  - CNN: MICN, ModernTCN, WaveNet
  - RNN/SSM: DeepAR, MambaSimple, S_Mamba, BiMamba, S4 (kernel-free, CPU-runnable)
  - Modern: FiLM, FreTS, Koopa, SOFTS, TimeKAN
  - 20 graph/spatiotemporal (Tier 2): STID, GWNet, STGCN, DCRNN, MTGNN, AGCRN,
    STNorm, StemGNN, STGODE, STAEformer, GTS, DGCRN, STDN, DFDGCN, STPGNN, D2STGNN,
    MegaCRN, HimNet, BigST, STWave
  - PoorOtterBob: MoFo, PHAT, BiST, MAGE, STOP, CauAir, AirCade
- **Three forecasting data settings** via `task.mode`: `time_series`,
  `spatiotemporal`, `covariate` (polymorphic mark handling in
  `src/models/_external/marks.py`).
- **Graph adjacency pipeline** — datasets expose `adj_mx`/`num_nodes`; the runner
  injects them into the model factory. Optional `[dataset.params] adj_norm`
  (symmetric/scaled Laplacian, GCN, transition matrices).
- **Datasets** — node-structured CauAir/synthetic_st, traffic graph bundles
  (METR-LA, PEMS-BAY, PEMS03/04/07/08 via `tsf dataset convert-traffic`), and plain-CSV
  configs (Exchange, ILI, Beijing-air, AQShunyi, AQWan, NN5, FRED-MD).
- **Metrics** — `corr`, `rse`, `wape`, `smape` (+ opt-in `mase`); **masked losses**
  (`masked_mae`/`mse`/`rmse`) for missing-value forecasting.
- **Pluggable training callbacks** — `[training.tricks]` curriculum / grad-clip /
  grad-accum, plus model aux-loss support.
- **`[evaluation] strategy="rolling"`** RollingForecast evaluation.
- **Profiling** — fit-time / inference-time columns in `performance.csv`.
- **Tools** — `tsf dataset convert-traffic`, `tsf result predictions`,
  `tsf dataset inspect`.
- **Agent Skills** — `experiments`, `characteristics` skills added (14 total),
  all with progressive disclosure (L1 frontmatter → L2 body → L3 docs/tools).
- **Governance** — MIT `LICENSE` (Copyright © 2026 Diaugeia.AI),
  `THIRD_PARTY_NOTICES.md` (per-model upstream + license for all 99), GitHub issue
  forms (new model / bug / feature), PR template, `CONTRIBUTING.md`.

### Changed

- The project README added a **Principles** section (Modern / Agentic /
  Reproducible / Open by default) and a license footer.
- Copyright re-declared to **Diaugeia.AI**.

### Removed

- **Stripped all non-forecasting branches** (imputation / anomaly detection /
  classification) and the `task_name` parameter from 7 TSLib-style models
  (Autoformer, FEDformer, TimesNet, TiDE, SegRNN, CrossLinear, MoFo). The project
  is now forecasting-only.
- Dropped M4 and imputation-as-a-task from the roadmap (out of scope).

### Notes

- `S_Mamba` / `BiMamba` / `S4` use a kernel-free selective scan (no `mamba_ssm`),
  so they run on CPU; GPU runs are recommended for speed.
- `PHAT` ships a paper reconstruction of its attention block (not an
  author-verified reproduction) — see `THIRD_PARTY_NOTICES.md`.

**Full diff:** https://github.com/Diaugeia/ModernTSF/compare/v0.1.0...v0.2.0

## [0.1.0]

Initial release — 31 models, TOML-driven config/registry/runner pipeline, classic
benchmark datasets, aggregation/ranking/plotting tools, and Agent Skills.

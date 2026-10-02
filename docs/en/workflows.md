# Workflows and architecture

ModernTSF has one flat model catalog, one shared-component area, one data
pipeline, and one verification route. Models and methods are peers.

```text
src/moderntsf/models/<slug>/              local model code, spec, and model card
src/moderntsf/models/_components/<name>/  reusable component code and card
src/moderntsf/data/                       dataset loaders and parameter schemas
dataset/                        local dataset bytes (not packaged)
catalog/datasets/<preset>/      generated readable dataset cards
configs/                        composable model, dataset, and run TOML
verification/                   manifest, generated index, per-model evidence
work_dirs/                      experiment checkpoints, metrics, and records
work_dirs/_research/<round>/    optional goals, events, prompts, and full logs
```

## Model runtime interface

Every model accepts the same call:

```python
forecast = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
```

- `x_enc`: observed values, normally `[batch, seq_len, channels]`.
- `x_mark_enc`: historical time/node covariates, or `None`.
- `x_dec`: observed decoder prefix plus the future placeholder.
- `x_mark_dec`: known future covariates, or `None`.

An implementation may ignore inputs its method does not use, but it must accept
the complete interface. Point outputs are `[batch, pred_len, targets]`; quantile
and distribution outputs append their parameter axis. `spec.py` contains only
the factory, validated parameter schema, preset path, task capabilities, shared
components, artifacts, optional training objective, and runtime fixture.
Descriptive facts belong in README.

The generic trainer owns batching, the four-input call, configured criterion,
callbacks, optimization, and validation. A paper-specific full training objective
must be declared as `ModelSpec.training_objective`; its adapter returns the
forecast and one finite scalar loss from a single pass. Do not advertise such an
objective as a capability or leave a `training_loss()` method that the runner does
not call. Validation and test metrics always use the configured observation loss.

`task.mode` is enforced before execution:

- `time_series`: multivariate values and optional calendar marks.
- `spatiotemporal`: node values with historical node/time covariates.
- `covariate`: node values plus known future covariates.

Inspect compatibility with `tsf model show <Name>` and `tsf dataset show <preset>`; both accept `--depth {0,1,2,3}` (see Components below).

## Add a model or method

The admission path is deliberately two-phase so a placeholder cannot become a
catalog entry.

1. Deduplicate the paper against `tsf model list --json` and `tsf model search`.
2. Read the paper and supplement. Locate official code when available, record its
   license, and pin a revision. Use it to clarify omitted implementation details;
   do not copy or import its model source.
3. Map every defining operation to an existing component, a justified new shared
   component, or a model-local block. Start with:

   ```bash
   uv run tsf component search "operation and tensor contract"
   uv run tsf component show <candidate>
   ```

4. Create an unregistered workspace:

   ```bash
   uv run tsf model scaffold \
     --name MyModel \
     --paper-title "Paper title" \
     --paper-url https://arxiv.org/abs/0000.00000 \
     --venue Conference --year 2026 \
     --code-url https://github.com/org/repo \
     --revision 0123456789abcdef \
     --license Apache-2.0 \
     --components revin,flatten_forecast_head \
     --params "enc_in:int,d_model:int=128"
   ```

   Omit all three code arguments when no official code exists. Use
   `--components none` only after matching found no equivalent. Select
   `--task-mode spatiotemporal` or `covariate` when required.
5. Replace every scaffold marker, implement locally, preserve useful paper
   equations/comments, complete the card, and add focused tests plus a declaration
   in `verification/models.toml`. Official code requires a reference-comparison
   test; absence of official code is recorded as `not-applicable`.
6. Admit the entry:

   ```bash
   uv run tsf model add --name MyModel
   ```

   Admission temporarily registers the model, runs unified verification, the
   focused model audit, strict runtime contracts, component audit, and repository
   audit. Registration is rolled back if any gate fails.

## Foundation models and artifacts

Foundation models remain ordinary flat model entries. Pretraining scale is not a
catalog category. A local architecture without released weights must say so in
its card and cannot claim zero-shot checkpoint behavior.

Released pretrained models use the thin boundary in `src/moderntsf/models/_foundation/`.
`FoundationModel` converts the canonical `[batch, time, channels]` input to the
official runtime's series batch and restores point or quantile output axes.
Chronos and TimesFM have direct adapters; Moirai accepts an official forecast
object from a Uni2TS-compatible environment. Provider packages are optional and
lazy. Construction is offline and receives an explicit local checkpoint path, so
the official cache is reused without copying or repackaging weights.

The flat model spec declares `inference-only`. The runner then skips optimizer,
training, and checkpoint creation while preserving the normal dataset and
evaluation paths. Upper-layer methods such as CoRA or retrieval augmentation stay
as separate flat entries and compose with a foundation runtime in an explicit
experiment rather than through a second registry.

Weights, tokenizers, or normalization statistics remain explicit runtime facts:

```python
from moderntsf.benchmark.registry.models import ModelArtifact

artifacts = (ModelArtifact(
    name="weights",
    url="https://host/repository/resolve/full-commit-or-release/weights.safetensors",
    revision="full-commit-or-release",
    sha256="<64 lowercase hex characters>",
    filename="weights.safetensors",
    required=True,
),)

def build_from_artifacts(cfg, params, paths):
    model = Model(...)
    model.load_checkpoint(paths["weights"])
    return model

SPEC = ModelSpec(
    ...,
    artifacts=artifacts,
    artifact_factory=build_from_artifacts,
)
```

ModernTSF never downloads them during construction. Inspect or explicitly fetch
an artifact before a run:

```bash
uv run tsf model artifacts MyFoundationModel
uv run tsf model artifacts MyFoundationModel --fetch weights
```

The cache defaults to the user cache directory and may be changed with
`MODERNTSF_CACHE`. A required artifact must exist and match SHA-256 before the
artifact-aware factory runs. Ordinary models keep the two-argument factory;
artifact-backed models explicitly receive a mapping of verified local paths.
Artifact tests, loading behavior, offline failure, and the exact checkpoint claim
belong in verification and the model card.

## Components

Components live only in `src/moderntsf/models/_components/<name>/`. Each directory has an
implementation, a catalog contract, focused tests, and a README card. The card is
curated (purpose and formula, origin and why it was cut at this boundary, every
public symbol with parameters and tensor shapes, equivalence evidence, variants,
when to use and not to use, related components); its API, import line, and
consumers are generated between the card markers. `tsf component audit` enforces
both parts.
Extract only mathematically and operationally equivalent behavior—matching names
or tensor rank is insufficient. Validate axes, normalization, masking, residual
order, initialization, state, outputs, gradients, and serialization.

```bash
uv run tsf component list
uv run tsf component search "patch forecast head"
uv run tsf component show flatten_forecast_head --depth 1
uv run tsf component audit
```

### Reading the catalog by depth

Models, components, and datasets share four depths, so you spend context only as
needed. Search returns one line per result (`name`, `kind`, `summary`, `tags`);
`show` takes `--depth` (default 1) and `--json` for structured output:

| Depth | Content |
| --- | --- |
| 0 | the one-line summary |
| 1 | front matter plus the interface and constraint sections |
| 2 | the full card |
| 3 | the source, config, test, and evidence paths to open |

```bash
uv run tsf catalog search "reversible normalization" --kind component
uv run tsf component show revin --depth 2
uv run tsf model show PatchTST --depth 3
```

Paper-specific variants stay inside the model package. Named model packages must
not import implementation code from another named model.

## Data

Data has three non-overlapping layers:

- `dataset/`: local files, downloads, and converted arrays; never code or cards.
- `src/moderntsf/data/`: executable loaders, base contracts, and Pydantic parameter schemas.
- `catalog/datasets/`: one generated README card per runnable dataset preset.

Fetch a published preset's files, pinned and checksum-verified, into `dataset/`
(see [the Hub page](hub.md#benchmark-data)):

```bash
uv run tsf dataset download --list
uv run tsf dataset download etth1
```

Use an existing CSV preset or create a loader-backed dataset:

```bash
uv run tsf dataset add --name my_data --pattern custom \
  --path ./dataset/my_data/my_data.csv --target OT
uv run tsf dataset inspect --config configs/datasets/my_data.toml
uv run tsf dataset analyze my_data      # model-selection profile (JSON + markdown)
uv run tsf dataset show my_data
uv run tsf dataset audit
```

`dataset analyze <preset>` (or `--path FILE`) profiles the training split (length,
missingness, scale, periods, seasonality, trend, forecastability, cross-channel
structure, outliers) plus train/validation shift, recommends lookback candidates, and
maps findings to catalog components and models. Test-split shift is labelled
diagnostic-only. Results go to `work_dirs/profiles/<name>/`.

`dataset prepare`, `convert-traffic`, and `gift-download` provide explicit
conversion/download operations; inspect their `--help` before writing. Scaling
must fit training data only, split boundaries must be stable, and graph/covariate
loaders must declare compatible task modes.

### PeMS traffic (UltraTraffic)

The UltraTraffic archive (hourly total flow per Caltrans PeMS station, four
districts, 2003–2023) is converted once into a local parquet store:

```bash
uv run tsf dataset download ultratraffic_ba_st                  # published slice, or
uv run tsf dataset convert-ultratraffic --archive TrafficCL.zip   # -> dataset/ultratraffic
```

`ultratraffic_st` (spatiotemporal, with calendar covariates) and
`ultratraffic_ts` (stations as channels) read the store; presets
`ultratraffic_{ba,la,sac,sb}_{st,ts}` load 2023 for Districts 4, 7, 3, and 8.
Parameters select `years`, the station policy across years (`last` or
`intersection`), and the continual-learning splits `cl_common` / `cl_added`
(stations retained from, or new relative to, the previous year; see
`ultratraffic_sb_cl`). The same store bootstraps the `traffic_pems_*`
real-time tracks.

## Experiments

A run TOML composes base, dataset, and model presets. Keep scientific choices in
config, not shell scripts:

```toml
extends = ["../base.toml", "../datasets/etth1.toml", "../models/DLinear.toml"]

[experiment]
description = "DLinear ETTh1 baseline"
random_seed = 42
work_dir = "./work_dirs"

[task]
mode = "time_series"
seq_len = 96
label_len = 0
pred_len = 96
features = "M"

[sweep]
experiment.random_seed = [0, 1, 2]
task.pred_len = [96, 192]
```

Preview the fully resolved matrix before spending compute:

```bash
uv run tsf inspect --config configs/runs/<run>.toml
uv run tsf run configs/runs/<run>.toml
```

Optional environment checks, tracking, budgets, and recovery are described in
[execution.md](execution.md). Ordinary runs need no policy or round.

Use `--jobs` and `--gpus` only after confirming resources. Each run preserves its
resolved config, seed, environment, checkpoints, raw metrics, and failures under
`work_dirs/`. Compare only cells with the same data split, preprocessing, horizon,
metric definition, and evaluation strategy. Result commands are discoverable with
`tsf result --help`; they aggregate, rank, plot, inspect predictions, and report.

For iterative work, create an optional research round instead of adding state to
the run config:

```bash
uv run tsf research start --task experiment --goal "Does RevIN improve MSE?" --max-runs 8
uv run tsf run configs/runs/<run>.toml --round <round-id>
uv run tsf research note <round-id> --kind decision --text "Keep the same seeds"
uv run tsf research status <round-id> completed --message "No improvement observed"
```

A round is only a small workspace containing `round.json`, append-only events,
complete command logs, and an optional rendered prompt. It does not alter model,
dataset, config, verification, or result contracts. Resolved runs consume the
declared budget atomically; failures remain visible. Existing Harnesses can be
rendered alone or started with a round:

```bash
uv run tsf agent task render experiment --set 'question=<question>'
uv run tsf agent task start autoresearch --set 'question=<question>' --json
```

`task start` prepares a round and a directly readable prompt. It deliberately
does not launch or message an external Agent; the current Harness owns execution.

## Verification and repository gates

There is one model verification structure:

```text
verification/models.toml
verification/index.json
verification/evidence/<Model>.json
```

Evidence checks paper structure, equations, construction, forward, backward,
finite outputs, active gradients, state-dict round trip, CPU, batch and sequence
boundaries, input contract, and reference comparison. The index is generated.

```bash
uv run tsf verify model DLinear
uv run tsf verify stale
uv run tsf verify all --jobs 8
uv run tsf verify index
uv run tsf model audit --summary
uv run tsf repo doctor --strict
uv run tsf repo audit
```

Paper-result reproduction is separate from code verification: reproduce datasets,
splits, preprocessing, optimization, seeds, metrics, and reported cells through
run configs, then state all deviations instead of treating a successful forward
pass as a reproduced paper result.

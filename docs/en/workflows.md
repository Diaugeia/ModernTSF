# Workflows and architecture

TSFLab has one flat model catalog, one shared-component area, one data
pipeline, and one verification route. Models and methods are peers.

```text
src/tsflab/models/<slug>/              local model code, spec, and model card
src/tsflab/models/_components/<name>/  reusable component code and card
src/tsflab/data/                       dataset loaders and parameter schemas
dataset/                        local dataset bytes (not packaged)
catalog/datasets/<preset>/      dataset cards: curated facts plus a generated runtime block
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
callbacks, optimization, and validation. By default a model trains on the
configured criterion plus an optional scalar `aux_loss` regularizer it sets during
`forward`. A paper-specific training loss is opt-in: declare
`ModelSpec.training_objective(model, batch, criterion)`, which receives a
`TrainingBatch` (`x`, `x_mark`, `dec_inp`, `y_mark`, `y`, `target`,
`batch.forecast(model)`, `batch.align(outputs)`) and the configured criterion, and
returns `(forecast_or_None, finite_scalar_loss)` from a single pass. It may extend
the criterion (card slot `loss:mse+local:<name>`) or replace it (`local:<name>`).
An optional `ModelSpec.training_setup(model, train_loader, *, pred_len, features)`
runs once before a fresh run to fit training-split state such as a label basis.
The objective is used only while training; validation, early stopping, and test
metrics always use `forward` with the configured observation loss. Objectives are
unsupported with `DataParallel`. Do not advertise one as a capability or leave a
`training_loss()` method that the runner does not call; a paper objective whose
inputs the runner cannot supply (for example a second historical window) stays off
and the model card's Differences says why.

`task.mode` is enforced before execution:

- `time_series`: multivariate values and optional calendar marks.
- `spatiotemporal`: node values with historical node/time covariates.
- `covariate`: node values plus known future covariates.

Inspect compatibility with `tsf model show <Name>` and `tsf dataset show <preset>`; both accept `--depth {0,1,2,3}` (see Components below).

## Add a model or method

The admission path is deliberately two-phase so a placeholder cannot become a
catalog entry.

1. Deduplicate the paper against `tsf model list --json` and `tsf catalog search --kind model`.
2. Read the paper and supplement. Locate official code when available, record its
   license, and pin a revision. Use it to clarify omitted implementation details;
   do not copy or import its model source.
3. Decide the retrieval-layer facts first, since readers find a model through them:
   a `tagline` (at most 120 characters), `tags` (with one architecture family), and
   the six-slot `composition` (below). Map every defining operation to an existing
   component, a justified new shared component, or a model-local block. Start with:

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
   equations/comments, complete the card (`tagline`, `tags`, `composition`, and a
   `## Key ideas` section before the generated block), and add focused tests plus a declaration
   in `verification/models.toml`. Official code requires a reference-comparison
   test; absence of official code is recorded as `not-applicable`.
6. Admit the entry:

   ```bash
   uv run tsf model add --name MyModel
   ```

   Admission registers the model, regenerates the cards, runs unified
   verification, the focused model audit, strict runtime contracts, the component
   audit, and the repository audit. Registration is rolled back if any gate fails.

## Foundation models and artifacts

Foundation models remain ordinary flat model entries. Pretraining scale is not a
catalog category. A local architecture without released weights must say so in
its card and cannot claim zero-shot checkpoint behavior.

Released pretrained models use the thin boundary in `src/tsflab/models/_foundation/`.
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
from tsflab.benchmark.registry.models import ModelArtifact

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

TSFLab never downloads them during construction. Inspect or explicitly fetch
an artifact before a run:

```bash
uv run tsf model artifacts MyFoundationModel
uv run tsf model artifacts MyFoundationModel --fetch weights
```

The cache defaults to the user cache directory and may be changed with
`TSFLAB_CACHE`. A required artifact must exist and match SHA-256 before the
artifact-aware factory runs. Ordinary models keep the two-argument factory;
artifact-backed models explicitly receive a mapping of verified local paths.
Artifact tests, loading behavior, offline failure, and the exact checkpoint claim
belong in verification and the model card.

## Components

Components live only in `src/tsflab/models/_components/<name>/`. Each directory has an
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

### Reading the catalog

Start at the entry view, then narrow. Each step costs more context than the last,
so stop as soon as the decision is made:

```bash
uv run tsf catalog                       # counts per kind, then the next commands
uv run tsf catalog search "reversible normalization" --kind component   # L0
uv run tsf model show PatchTST           # L1
uv run tsf component show revin --depth 2   # L2
uv run tsf model show PatchTST --depth 3    # L3
```

| Depth | Content |
| --- | --- |
| 0 | one line: `name`, `kind`, `summary`, `tags` (what search returns) |
| 1 | front matter plus the interface and constraint sections (default) |
| 2 | the full card |
| 3 | the source, config, test, and evidence paths to open |

`catalog search` ranks all three kinds; `--kind` restricts it, `--capability`
(repeatable) filters models, `--limit` caps results, and `--json` gives
structured output. The per-kind `search` and `show` commands return the same lines
and cards.

What each card records:

- **Model:** a `tagline`, `tags` (one is the architecture family), and a
  `composition` of six slots, `normalization`, `decomposition`, `temporal`,
  `channel`, `head`, `loss`, each `component:<name>`, `local:<block>`,
  `loss:<name>`, or `none`. A `## Key ideas` section lists what is distinctive. The
  paper, venue, and (when it exists) official code, pinned revision, and license
  come with it, and the body maps operations to local code and lists differences
  from the paper and official code.
- **Component:** `summary`, `category`, `input`/`output` shapes, `origin`, `tags`,
  and the sections Purpose, Origin and granularity, Interface (every public symbol
  with shapes and state), Invariants and equivalence evidence, Variants and
  options, When to use and when not to use, and Related components.
- **Dataset:** domain, source, citation, license, redistribution, frequency, time
  span, length, channels, target, missing values, the TSFLab `protocol`, the
  `literature_protocol` when the published one differs, lookbacks and horizons,
  split, and whether the statistics were measured or source-reported. See Data.

Paper-specific variants stay inside the model package. Named model packages must
not import implementation code from another named model.

## Data

Data has three non-overlapping layers:

- `dataset/`: local files, downloads, and converted arrays; never code or cards.
- `src/tsflab/data/`: executable loaders, base contracts, and Pydantic parameter schemas.
- `catalog/datasets/`: one README card per runnable dataset preset, plus one family
  card for GIFT-Eval. Each card pairs curated facts (domain, source, license,
  statistics, protocol, pitfalls) with a generated runtime block.

There are 99 dataset presets: 75 conventional `time_series` presets (including the
GIFT-Eval series) and 24 spatiotemporal or covariate presets; 16 presets under
`rt/` are frozen releases of the real-time tracks. Every dataset has
exactly one TSFLab protocol, stated in its card (chronological split, scaling fitted
on the training split only, lookbacks and horizons), so results on a dataset are
comparable across models. Where the literature uses a different protocol, the card
records it separately as `literature_protocol`. `configs/fixtures/` holds smoke and
synthetic test inputs; they are not datasets and have no cards.

Fetch a published preset's files, pinned and checksum-verified, into `dataset/`
(see [the Hub page](hub.md#benchmark-data)):

```bash
uv run tsf dataset download --list
uv run tsf dataset download etth1
```

Find and read datasets by what they are, not only by name. Search matches the
card's domain, tags, frequency, and source; `show` returns the card's facts:

```bash
uv run tsf dataset search hourly electricity
uv run tsf dataset show etth1       # preset record plus card facts
uv run tsf dataset audit            # required facts, no placeholders, generated block current
```

Card front matter (short facts: `summary`, `domain`, `tags`, `source`, `license`,
`redistribution`, `frequency`, `time_span`, `length`, `channels`, `target`,
`missing_values`, `protocol`, `literature_protocol`, `seq_lens`, `pred_lens`, `split`, `stats_basis`,
`related`, and optionally `realtime_track`) is the quick reference; the body adds provenance, statistics, standard
protocol, and known pitfalls. `license: "unknown"` means no explicit terms were
found, not that redistribution is allowed. `stats_basis` says whether numbers
were measured from local files or reported by the source. `tsf repo cards`
rewrites only the marked runtime block and the generated front-matter keys.

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
diagnostic-only. Results go to `work_dirs/profiles/<name>/` (`--out` changes it,
`--json` prints the profile). For a file that has no preset, pass `--path FILE`
with `--split-ratio TRAIN VAL TEST` and optionally `--freq`.

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

## AutoResearch

AutoResearch asks which design suits a dataset and tests it, instead of running a
fixed benchmark. It composes three public pieces and needs no new state beyond an
optional research round.

1. **Profile the data.** `tsf dataset analyze <preset>` reports the training-split
   statistics and the recommended lookbacks and catalog options (see Data).
2. **Fill the slot grid.** A model is described by six slots, `normalization`,
   `decomposition`, `temporal`, `channel`, `head`, and `loss`, plus at most one
   bounded free-form block for something the catalog cannot express. Options are
   `component:<name>`, `model:<Name>` (a donor design whose idea is borrowed;
   models never import peers), or `loss:<name>`. Start from the best baseline,
   screen one slot at a time, then combine the best compatible winners.
3. **Validate the recombination.** Write a TOML spec and validate it:

   ```toml
   name = "SeasonalRevLinear"
   summary = "RevIN and decomposition around a channel-wise linear map."
   hypothesis = "strong seasonality: a period-aware split beats RLinear at equal lookback."
   parents = ["RLinear", "DLinear"]
   [slots]
   normalization = ["component:revin"]
   decomposition = ["component:series_decomposition"]
   temporal = ["component:channel_wise_linear"]
   channel = ["local:independent"]      # local:independent | local:individual | local:mixing
   head = ["component:flatten_forecast_head"]
   loss = ["loss:mae"]
   [params]                             # optional: hidden, n_layers, n_heads, patch_len, stride, kernel_size, dropout
   hidden = 32
   ```

   ```bash
   uv run tsf component compose spec.toml
   ```

   Without flags `compose` writes nothing. It checks that components, models, and the
   loss are real, that component symbols import, and that the free-form budget (two
   blocks, 120 lines each) holds, then reports whether the slots are **executable**
   by the slot adapters (`executable: yes (point output)` or `NOT EXECUTABLE: <reason>`,
   for example `mixer_block` needs `channel = "mixing"`).

   Executable options, one per slot (absent slots default to `none` / `independent` /
   `flatten_forecast_head` / `mse`):

   | slot | options |
   | --- | --- |
   | normalization | `none`, `component:revin`, `component:last_value_center` |
   | decomposition | `none`, `component:series_decomposition` (`wavelet` is rejected: subbands change length) |
   | temporal | `local:linear`, `component:channel_wise_linear`, `component:tst_transformer` (alias `component:patchtst`), `component:mamba`, `component:gated_dilated_conv`, `component:mixer_block` |
   | channel | `local:independent` (shared weights), `local:individual` (per-channel head weights), `local:mixing` (only with `mixer_block`) |
   | head | `component:flatten_forecast_head` (point), `component:quantile_head` (needs `loss:quantile`), `component:gaussian_parameter_head` (needs `loss:nll_gaussian`) |

4. **Make it runnable and run it.** `--write-config` turns the validated spec into a
   run config that extends a base, a dataset, and the `Composed` model with the slot
   parameters (the point model; quantile and Gaussian heads need `--register` below):

   ```bash
   uv run tsf component compose spec.toml --write-config configs/runs/auto_seasonal.toml \
     --dataset etth1 --enc-in 7 --pred-len 96 --work-dir work_dirs/round1
   # tiny CPU check on the fixture dataset instead:
   uv run tsf component compose spec.toml --write-config /tmp/auto.toml \
     --dataset configs/fixtures/smoke.toml --enc-in 6 --pred-len 12 --smoke
   uv run tsf smoke --config /tmp/auto.toml        # or: uv run tsf run configs/runs/auto_seasonal.toml --round <id>
   ```

   `--dataset` is a TOML path or a `configs/datasets/<name>.toml` preset name.

5. **Read the bar to beat.** The board merges the committed leaderboard with local run
   records and prints compact lines, best first (`rank model metric mae/mse H n source`):

   ```bash
   uv run tsf result board --dataset ETTh1 --horizon 96 --top 5
   uv run tsf result board --dataset weather --records work_dirs/round1 --json
   ```

   Local rows list their `seq` and `epochs`; a short or smoke run is not comparable
   with a published row. Slot-assignment runs of `Composed` are labelled by their
   slots, for example `Composed[revin/none/linear/independent]`.

6. **Register the winner.** After confirmation seeds, scaffold a dedicated catalog
   model whose card records the composition and hypothesis as provenance, then admit
   it through the normal gates:

   ```bash
   uv run tsf component compose spec.toml --register SeasonalRevLinear --dry-run   # lists the files
   uv run tsf component compose spec.toml --register SeasonalRevLinear
   uv run tsf model add --name SeasonalRevLinear   # verification, audits, catalog entry; rolls back on failure
   ```

   `--register` writes `src/tsflab/models/<slug>/` (model, spec, card, preset) plus a
   `verification/models.toml` entry; the generated model fixes the slots, so its
   quantile or Gaussian head declares the matching output capability. Combine
   `--register NAME --write-config PATH` to get a config that runs the new model.

Many wins need no code: change `training.loss`, `task.seq_len`, or a model
parameter in the run config. Register a new model only after it beats the baselines
with confirmation seeds, through the normal path in
[Add a model or method](#add-a-model-or-method).

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

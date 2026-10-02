# Projects and the Hugging Face Hub

← [Documentation index](README.md)

## Standalone projects

`tsf init <dir>` scaffolds a project that depends on the installed package
instead of a checkout:

```
<dir>/
├── configs/runs/example.toml   # extends tsflab://configs/...
├── dataset/                    # local data (ignored)
├── pyproject.toml              # depends on the package's hub extra
└── README.md
```

Run configs may extend catalog presets with `tsflab://` paths, which resolve
against the checkout or the installed package's read-only assets:

```toml
extends = [
    "tsflab://configs/base.toml",
    "tsflab://configs/datasets/etth1.toml",
    "tsflab://configs/models/DLinear.toml",
]
```

Outputs are written to the project's own `work_dirs/`.

## One address for every published asset

Remote weights, checkpoints, and data use one pinned URI form:

```
hf://[datasets/|spaces/]<owner>/<repo>@<revision>/<path>
```

The revision is mandatory, so a URI always names the same bytes. Model
`ModelArtifact` declarations accept `hf://` URIs next to `https://` and
`file://`, and every download is SHA-256 verified into `$TSFLAB_CACHE`
(default `~/.cache/tsflab`). `HF_TOKEN` is sent only to the Hugging Face
endpoint, so private repositories work; `HF_ENDPOINT` overrides the endpoint.

## Published repositories

| Repository | Type | Contents |
| --- | --- | --- |
| `Diaugeia/TSFLab-Static` | dataset | files behind the dataset presets, laid out as `dataset/` |
| `Diaugeia/TSFLab-RealTime` | dataset | append-only real-time track panels, one commit per release |
| `Diaugeia/TSFLab-Weights` | model | trained weights bundles |
| `Diaugeia/TSFLab` | space | the static leaderboard site |

A fork or personal mirror sets `TSFLAB_HUB_OWNER` to publish under another
namespace instead of passing `--repo` to every command. Maintainers create them, with their cards, through
`uv run tsf hub init [--migrate-legacy]`; `--migrate-legacy` renames the former
TSEval repositories so their old addresses redirect.

## Benchmark data

`configs/hub/datasets.json` pins every published data file by commit and
SHA-256. Presets download into the local `dataset/` root, verified:

```bash
uv run tsf dataset download --list          # presets with published files
uv run tsf dataset download etth1 weather   # or --all
uv run tsf dataset download --check         # every pinned file still resolves
```

UltraTraffic presets fetch only their region, variant, and years. Maintainers
publish local files, which updates the manifest to commit with the change:

```bash
uv run tsf dataset publish etth1 etth2 [--create]
uv run tsf dataset publish --path ultratraffic      # a whole store, every year
```

Each dataset keeps its source license; publish only data whose terms allow
redistribution.

## Weights bundles

A finished run is published as one bundle directory at
`<dataset>/<model>/<run_id>/` in a model repository (default
`Diaugeia/TSFLab-Weights`):

| File | Contents |
| --- | --- |
| `manifest.json` | identity, shapes, metrics, provenance, per-file SHA-256 |
| `model.safetensors` | the best checkpoint's `state_dict` |
| `record.json` | the run's TSF-Core `RunRecord` |
| `README.md` | model card rendered from the manifest |

```bash
uv run tsf hub pack <run_id>                      # local bundle under work_dirs/_bundles/
uv run tsf hub push <run_id> --repo <owner>/<repo> [--create] [--public]
uv run tsf hub list --repo <owner>/<repo> --dataset weather
uv run tsf hub pull hf://<owner>/<repo>@<revision>/weather/DLinear/<run_id>
```

`push` uploads only the run it is given, creates private repositories unless
`--public` is passed, and prints the bundle URI pinned to the new commit.

```python
from tsflab import hub

state_dict, manifest = hub.load_state_dict(
    "hf://Diaugeia/TSFLab-Weights@<revision>/weather/DLinear/<run_id>"
)
```

Packing, loading, and uploading need the package's `hub` extra
(`huggingface_hub`, `safetensors`); plain downloads use only the standard library.

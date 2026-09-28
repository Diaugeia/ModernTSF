# Projects and the Hugging Face Hub

← [Documentation index](README.md)

## Standalone projects

`tsf init <dir>` scaffolds a project that depends on the installed package
instead of a checkout:

```
<dir>/
├── configs/runs/example.toml   # extends moderntsf://configs/...
├── dataset/                    # local data (ignored)
├── pyproject.toml              # depends on the package's hub extra
└── README.md
```

Run configs may extend catalog presets with `moderntsf://` paths, which resolve
against the checkout or the installed package's read-only assets:

```toml
extends = [
    "moderntsf://configs/base.toml",
    "moderntsf://configs/datasets/etth1.toml",
    "moderntsf://configs/models/DLinear.toml",
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
`file://`, and every download is SHA-256 verified into `$MODERNTSF_CACHE`
(default `~/.cache/moderntsf`). `HF_TOKEN` is sent only to the Hugging Face
endpoint, so private repositories work; `HF_ENDPOINT` overrides the endpoint.

## Weights bundles

A finished run is published as one bundle directory at
`<dataset>/<model>/<run_id>/` in a model repository (default
`Diaugeia/ModernTSF-Weights`):

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
from moderntsf import hub

state_dict, manifest = hub.load_state_dict(
    "hf://Diaugeia/ModernTSF-Weights@<revision>/weather/DLinear/<run_id>"
)
```

Packing, loading, and uploading need the package's `hub` extra
(`huggingface_hub`, `safetensors`); plain downloads use only the standard library.

"""Upload bundles to, and list bundles in, a Hugging Face repository.

Requires ``huggingface_hub`` (``tsflab[hub]``) and a write token in
``HF_TOKEN`` or the local ``huggingface-cli login`` credentials.
"""

from __future__ import annotations

from pathlib import Path

from tsflab.hub.bundle import MANIFEST
from tsflab.hub.uri import DEFAULT_OWNER, HubURI, default_repo

DEFAULT_WEIGHTS_REPO = default_repo("TSFLab-Weights")


def _api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required; install `tsflab[hub]`") from exc
    return HfApi()


def push(bundle_dir: Path, path_in_repo: str, repo_id: str = DEFAULT_WEIGHTS_REPO,
         *, private: bool = True, create: bool = False) -> HubURI:
    """Upload one bundle directory and return its URI pinned to the new commit."""
    api = _api()
    if create:
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(bundle_dir),
        path_in_repo=path_in_repo,
        commit_message=f"add {path_in_repo}",
    )
    return HubURI(repo_id=repo_id, revision=commit.oid, path=path_in_repo)


def list_bundles(repo_id: str = DEFAULT_WEIGHTS_REPO, revision: str = "main",
                 dataset: str | None = None, model: str | None = None) -> list[str]:
    """Return ``<dataset>/<model>/<run_id>`` directories present in ``repo_id``."""
    files = _api().list_repo_files(repo_id, repo_type="model", revision=revision)
    bundles = []
    for name in files:
        parts = name.split("/")
        if len(parts) != 4 or parts[-1] != MANIFEST:
            continue
        if (dataset and parts[0] != dataset) or (model and parts[1] != model):
            continue
        bundles.append("/".join(parts[:3]))
    return sorted(bundles)


def _card(front: dict[str, object], body: str) -> str:
    lines = ["---"] + [f"{key}: {value}" for key, value in front.items()] + ["---", "", body.strip(), ""]
    return "\n".join(lines)


_SOURCE = "https://github.com/Diaugeia/TSFLab"


def repository_plan(owner: str = DEFAULT_OWNER) -> list[dict[str, str]]:
    """Return the published repositories: id, type, and README card."""
    return [
        {"repo_id": f"{owner}/TSFLab-Static", "repo_type": "dataset", "card": _card(
            {"license": "other", "pretty_name": "TSFLab Static Benchmarks"},
            f"""
# TSFLab static benchmark data

Files behind the TSFLab dataset presets, laid out exactly as the local
`dataset/` root. Each file is pinned by commit and SHA-256 in
`configs/hub/datasets.json` of [TSFLab]({_SOURCE}); fetch them with

```bash
tsf dataset download <preset>   # or --all
```

Every dataset keeps the license of its original source; see its TSFLab
dataset card for provenance and terms.""")},
        {"repo_id": f"{owner}/TSFLab-RealTime", "repo_type": "dataset", "card": _card(
            {"license": "other", "pretty_name": "TSFLab Real-Time Panels"},
            f"""
# TSFLab real-time panels

Append-only panels for the weekly rolling tracks of [TSFLab]({_SOURCE})
(PeMS traffic, CSI 300 stocks, air quality). One directory per track; each
weekly release is one commit, so every round is reproducible from its pinned
revision. Maintained by the `realtime-weekly` workflow.""")},
        {"repo_id": f"{owner}/TSFLab-Weights", "repo_type": "model", "card": _card(
            {"license": "mit", "library_name": "tsflab"},
            f"""
# TSFLab trained weights

Checksummed safetensors bundles at `<dataset>/<model>/<run_id>/`, published with
`tsf hub push` and loaded with `tsf hub pull hf://{owner}/TSFLab-Weights@<revision>/...`.
Source: [TSFLab]({_SOURCE}).""")},
        {"repo_id": f"{owner}/TSFLab", "repo_type": "space", "card": _card(
            {"title": "TSFLab Leaderboard", "emoji": '"📈"', "colorFrom": "gray",
             "colorTo": "yellow", "sdk": "static", "pinned": "false", "license": "mit"},
            f"""
# TSFLab Leaderboard

Static leaderboard auto-deployed by the `web-deploy` workflow of
[TSFLab]({_SOURCE}).""")},
    ]


# Former TSEval repositories; moving them keeps their old URLs redirecting here.
LEGACY_NAMES = {"TSFLab-Static": "TSEval-Static", "TSFLab-RealTime": "TSEval-RealTime",
                "TSFLab": "TSEval"}


def init_repositories(owner: str = DEFAULT_OWNER, *, private: bool = False,
                      migrate: bool = False, dry_run: bool = False) -> list[str]:
    """Create any missing published repositories and write their cards (idempotent).

    With ``migrate``, a missing repository whose TSEval predecessor exists is
    renamed from it instead, so the Hub redirects the old address.
    """
    plan = repository_plan(owner)
    api = None if dry_run else _api()
    created = []
    for item in plan:
        kind = item["repo_type"]
        legacy = LEGACY_NAMES.get(item["repo_id"].split("/", 1)[1]) if migrate else None
        if dry_run:
            action = f"create (or rename from {owner}/{legacy})" if legacy else "create"
            created.append(f"{action} {kind}:{item['repo_id']}")
            continue
        if legacy and not api.repo_exists(item["repo_id"], repo_type=kind) \
                and api.repo_exists(f"{owner}/{legacy}", repo_type=kind):
            api.move_repo(f"{owner}/{legacy}", item["repo_id"], repo_type=kind)
        extra = {"space_sdk": "static"} if kind == "space" else {}
        api.create_repo(item["repo_id"], repo_type=kind, private=private, exist_ok=True, **extra)
        api.upload_file(path_or_fileobj=item["card"].encode("utf-8"), path_in_repo="README.md",
                        repo_id=item["repo_id"], repo_type=kind,
                        commit_message="docs: repository card")
        created.append(f"{kind}:{item['repo_id']}")
    return created

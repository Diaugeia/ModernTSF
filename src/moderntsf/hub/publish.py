"""Upload bundles to, and list bundles in, a Hugging Face repository.

Requires ``huggingface_hub`` (``modern-tsf[hub]``) and a write token in
``HF_TOKEN`` or the local ``huggingface-cli login`` credentials.
"""

from __future__ import annotations

from pathlib import Path

from moderntsf.hub.bundle import MANIFEST
from moderntsf.hub.uri import HubURI

DEFAULT_WEIGHTS_REPO = "Diaugeia/ModernTSF-Weights"


def _api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required; install `modern-tsf[hub]`") from exc
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

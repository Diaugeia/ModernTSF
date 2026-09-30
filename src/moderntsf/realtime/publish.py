"""Mirror track panels to a Hugging Face dataset, one pinned revision per release."""

from __future__ import annotations

from pathlib import Path

from moderntsf.hub.fetch import sha256_file
from moderntsf.hub.uri import default_repo
from moderntsf.realtime.store import PanelStore

DEFAULT_DATASET_REPO = default_repo("ModernTSF-RealTime")


def _api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("publishing needs `modern-tsf[hub]`") from exc
    return HfApi()


def push_release(store: PanelStore, repo_id: str = DEFAULT_DATASET_REPO, *, create: bool = False) -> str:
    """Upload the track directory and return the new commit revision."""
    api = _api()
    if create:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    release = store.latest_release()
    commit = api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(store.directory),
        path_in_repo=store.track, commit_message=f"{store.track}: release {release['version']}",
    )
    return commit.oid


def pull_track(store: PanelStore, repo_id: str = DEFAULT_DATASET_REPO, revision: str = "main") -> bool:
    """Download a published track into the local store; ``False`` if it does not exist yet."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    try:
        snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                          allow_patterns=[f"{store.track}/*", f"{store.track}/**/*"],
                          local_dir=str(store.directory.parent))
    except (RepositoryNotFoundError, EntryNotFoundError):
        return False
    return store.exists


def file_digests(directory: Path) -> dict[str, str]:
    return {str(p.relative_to(directory)): sha256_file(p) for p in sorted(directory.rglob("*")) if p.is_file()}

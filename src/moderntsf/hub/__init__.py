"""Unified access to published ModernTSF assets on the Hugging Face Hub.

Every remote asset — pretrained foundation weights, trained checkpoints,
datasets — is addressed by one pinned URI form::

    hf://[datasets/|spaces/]<owner>/<repo>@<revision>/<path>

Downloading uses only the standard library; packing, loading, and uploading
checkpoints need ``modern-tsf[hub]``.
"""

from moderntsf.hub.bundle import bundle_path, find_run, load_manifest, load_state_dict, pack
from moderntsf.hub.fetch import default_cache_root, fetch, resolve_url
from moderntsf.hub.publish import DEFAULT_WEIGHTS_REPO, list_bundles, push
from moderntsf.hub.uri import HubURI, is_hub_uri, parse

__all__ = [
    "DEFAULT_WEIGHTS_REPO",
    "HubURI",
    "bundle_path",
    "default_cache_root",
    "fetch",
    "find_run",
    "is_hub_uri",
    "list_bundles",
    "load_manifest",
    "load_state_dict",
    "pack",
    "parse",
    "push",
    "resolve_url",
]

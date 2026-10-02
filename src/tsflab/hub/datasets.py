"""Publish and download benchmark dataset files through one pinned manifest.

``configs/hub/datasets.json`` maps every published file (a path relative to the
local ``dataset/`` root) to the commit that holds it and its SHA-256. A preset's
files are selected from its config ``path``; UltraTraffic presets narrow that to
their region, variant, and years. Downloads use only the standard library;
publishing requires ``huggingface_hub`` and explicit authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tomllib

from tsflab.hub.fetch import download, sha256_file
from tsflab.hub.uri import default_repo
from tsflab.tsf_core.paths import is_packaged_root, repository_root

DEFAULT_STATIC_REPO = default_repo("TSFLab-Static")
MANIFEST_RELATIVE = Path("configs") / "hub" / "datasets.json"
_DATASET_PREFIX = "./dataset/"


@dataclass(frozen=True)
class Selection:
    """The files one dataset preset needs, as a predicate on relative paths."""

    preset: str
    base: str
    region: str | None = None
    variant: str | None = None
    years: tuple[int, ...] = ()

    def matches(self, relative: str) -> bool:
        if not self.region:
            return relative == self.base or relative.startswith(self.base + "/")
        root = self.base + "/"
        if relative == root + "manifest.json":
            return True
        region = f"{root}{self.region}/"
        if not relative.startswith(region):
            return False
        rest = relative[len(region):]
        if "/" not in rest:  # region-level sidecars (sensor change logs)
            return True
        if not self.variant:
            return True
        from tsflab.data.ultratraffic_store import _FILES

        template = _FILES[self.variant]
        if not self.years:
            return rest.startswith(template.split("/", 1)[0] + "/")
        stems = {template.format(year=year).rsplit(".", 1)[0] for year in self.years}
        return rest.rsplit(".", 1)[0] in stems


def selection(preset: str, root: Path | None = None) -> Selection:
    """Return the file selection for one dataset preset config."""
    config = (root or repository_root()) / "configs" / "datasets" / f"{preset}.toml"
    if not config.is_file():
        raise FileNotFoundError(f"unknown dataset preset {preset!r}")
    dataset = tomllib.loads(config.read_text(encoding="utf-8")).get("dataset", {})
    path = str(dataset.get("path", ""))
    if not path.startswith(_DATASET_PREFIX):
        raise ValueError(f"preset {preset!r} has no downloadable files (path={path!r})")
    base = path[len(_DATASET_PREFIX):].strip("/")
    params = dataset.get("params", {})
    region = params.get("region")
    return Selection(
        preset=preset,
        base=base,
        region=region,
        variant=params.get("variant") if region else None,
        years=tuple(int(year) for year in params.get("years", ())) if region else (),
    )


def manifest_path(root: Path | None = None) -> Path:
    return (root or repository_root()) / MANIFEST_RELATIVE


def load_manifest(root: Path | None = None) -> dict:
    path = manifest_path(root)
    if not path.is_file():
        return {"schema_version": 1, "repo": DEFAULT_STATIC_REPO, "files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def published_files(preset: str, manifest: dict, root: Path | None = None) -> dict[str, dict]:
    chosen = selection(preset, root)
    return {name: entry for name, entry in manifest["files"].items() if chosen.matches(name)}


def available_presets(root: Path | None = None) -> list[str]:
    """Return presets with at least one published file."""
    base = root or repository_root()
    manifest = load_manifest(base)
    names = []
    for config in sorted((base / "configs" / "datasets").glob("*.toml")):
        try:
            if published_files(config.stem, manifest, base):
                names.append(config.stem)
        except (FileNotFoundError, ValueError):
            continue
    return names


def fetch_preset(preset: str, data_root: Path = Path("dataset"), root: Path | None = None) -> list[Path]:
    """Download and verify every published file of ``preset`` under ``data_root``."""
    manifest = load_manifest(root)
    files = published_files(preset, manifest, root)
    if not files:
        raise FileNotFoundError(
            f"preset {preset!r} is not published; see `tsf dataset download --list`"
        )
    repo = manifest["repo"]
    written = []
    for name, entry in sorted(files.items()):
        uri = f"hf://datasets/{repo}@{entry['revision']}/{name}"
        written.append(download(uri, data_root / name, entry["sha256"]))
    return written


def local_files(preset: str, data_root: Path = Path("dataset"), root: Path | None = None,
                *, chosen: Selection | None = None) -> list[str]:
    chosen = chosen or selection(preset, root)
    start = data_root / chosen.base
    candidates = [start] if start.is_file() else sorted(p for p in start.rglob("*") if p.is_file())
    names = [p.relative_to(data_root).as_posix() for p in candidates]
    return [name for name in names if chosen.matches(name) and not Path(name).name.startswith(".")]


def publish_presets(presets: list[str], data_root: Path = Path("dataset"),
                    repo_id: str = DEFAULT_STATIC_REPO, *, paths: tuple[str, ...] = (),
                    create: bool = False, private: bool = False,
                    root: Path | None = None) -> dict[str, str]:
    """Upload local files (one commit per preset or raw ``paths`` subtree) and pin them.

    ``paths`` are directories or files relative to ``data_root`` published in
    full, for example a complete store beyond what any preset reads.
    """
    if is_packaged_root(root or repository_root()):
        raise RuntimeError("publishing datasets requires a TSFLab checkout")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("publishing needs `tsflab[hub]`") from exc
    api = HfApi()
    if create:
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    manifest = load_manifest(root)
    if manifest["files"] and manifest["repo"] != repo_id:
        raise ValueError(f"the manifest already pins files in {manifest['repo']}; "
                         "publish to that repository or start a new manifest")
    manifest["repo"] = repo_id
    revisions = {}
    targets = [(preset, None) for preset in presets]
    targets += [(path.strip("/"), Selection(preset=path, base=path.strip("/"))) for path in paths]
    for preset, chosen in targets:
        names = local_files(preset, data_root, root, chosen=chosen)
        if not names:
            raise FileNotFoundError(f"no local files for {preset!r} under {data_root}")
        digests = {name: sha256_file(data_root / name) for name in names}
        pending = [
            name for name in names
            if manifest["files"].get(name, {}).get("sha256") != digests[name]
        ]
        if not pending:
            revisions[preset] = "unchanged"
            continue
        commit = api.upload_folder(
            repo_id=repo_id, repo_type="dataset", folder_path=str(data_root),
            allow_patterns=pending, commit_message=f"{preset}: {len(pending)} file(s)",
        )
        for name in pending:
            manifest["files"][name] = {
                "revision": commit.oid, "sha256": digests[name],
                "size": (data_root / name).stat().st_size,
            }
        revisions[preset] = commit.oid
    manifest["files"] = dict(sorted(manifest["files"].items()))
    target = manifest_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return revisions


def check_manifest(root: Path | None = None) -> list[str]:
    """Return problems with pinned files: unreachable, or size differs (HEAD only)."""
    from urllib.error import URLError
    from urllib.request import Request, urlopen

    from tsflab.hub.fetch import resolve_url

    manifest = load_manifest(root)
    problems = []
    for name, entry in manifest["files"].items():
        url = resolve_url(f"hf://datasets/{manifest['repo']}@{entry['revision']}/{name}")
        try:
            # HF answers HEAD on /resolve/ with the file size before any LFS redirect.
            request = Request(url, method="HEAD")
            request.add_header("Accept-Encoding", "identity")
            with urlopen(request, timeout=30) as response:
                size = response.headers.get("X-Linked-Size") or response.headers.get("Content-Length")
        except URLError as exc:
            problems.append(f"{name}: unreachable ({exc})")
            continue
        if size is not None and int(size) != entry["size"]:
            problems.append(f"{name}: size {size} != pinned {entry['size']}")
    return problems


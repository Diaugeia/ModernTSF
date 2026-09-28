"""Explicit, checksum-verified artifact storage for model weights and tokenizers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

from moderntsf.benchmark.registry.models import ModelArtifact, ModelSpec


def default_cache_root() -> Path:
    """Return the model artifact cache without creating it."""
    override = os.environ.get("MODERNTSF_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".cache" / "moderntsf"


def artifact_path(spec: ModelSpec, artifact: ModelArtifact, cache_root: Path) -> Path:
    """Return a traversal-safe cache path for one pinned artifact."""
    revision = "".join(c if c.isalnum() or c in "._-" else "_" for c in artifact.revision)
    return cache_root / "models" / spec.module.rsplit(".", 1)[-1] / revision / artifact.filename


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_status(spec: ModelSpec, cache_root: Path | None = None) -> list[dict[str, object]]:
    """Describe declared artifacts and validate files already present in the cache."""
    root = cache_root or default_cache_root()
    records = []
    for artifact in spec.artifacts:
        path = artifact_path(spec, artifact, root)
        present = path.is_file()
        actual = sha256_file(path) if present else None
        records.append(
            {
                "name": artifact.name,
                "url": artifact.url,
                "revision": artifact.revision,
                "sha256": artifact.sha256,
                "filename": artifact.filename,
                "required": artifact.required,
                "path": str(path),
                "present": present,
                "verified": present and actual == artifact.sha256,
                "actual_sha256": actual,
            }
        )
    return records


def require_artifacts(
    spec: ModelSpec, cache_root: Path | None = None
) -> dict[str, Path]:
    """Return verified artifact paths or fail before model construction."""
    records = artifact_status(spec, cache_root)
    failures = [
        record for record in records if record["required"] and not record["verified"]
    ]
    if failures:
        names = ", ".join(str(record["name"]) for record in failures)
        raise FileNotFoundError(
            f"model {spec.name!r} requires missing or invalid artifact(s): {names}. "
            f"Inspect with `tsf model artifacts {spec.name}` and fetch each artifact "
            "explicitly with `--fetch <name>`."
        )
    return {
        str(record["name"]): Path(str(record["path"]))
        for record in records
        if record["verified"]
    }


def fetch_artifact(
    spec: ModelSpec,
    artifact: ModelArtifact,
    cache_root: Path | None = None,
) -> Path:
    """Download one explicitly requested artifact and atomically verify it."""
    root = cache_root or default_cache_root()
    destination = artifact_path(spec, artifact, root)
    if destination.is_file() and sha256_file(destination) == artifact.sha256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{artifact.filename}.", suffix=".part", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with urlopen(artifact.url, timeout=60) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        actual = sha256_file(temporary)
        if actual != artifact.sha256:
            raise ValueError(
                f"artifact {artifact.name!r} checksum mismatch: "
                f"expected {artifact.sha256}, got {actual}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination

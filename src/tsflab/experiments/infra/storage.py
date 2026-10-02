"""Atomic local records, crash-safe locks, and reproducibility fingerprints."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def file_lock(path: Path, *, blocking=True):
    """POSIX advisory lock released by the kernel even after a process crash."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield stream.fileno()
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def __getattr__(name):
    # Compatibility for callers of the original storage module. New consumers
    # use fingerprints explicitly; basic persistence imports no scientific code.
    if name in {"dataset_fingerprint", "code_fingerprint", "dependency_fingerprint"}:
        from tsflab.experiments.infra import fingerprints

        return getattr(fingerprints, name)
    raise AttributeError(name)

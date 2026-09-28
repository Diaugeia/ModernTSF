"""Storage inspection and explicit cleanup of obsolete managed epoch files."""

from pathlib import Path
import re

from moderntsf.benchmark.infra.storage import file_lock


def storage_status(directory, policy=None):
    from moderntsf.benchmark.infra.policy import Storage

    options = getattr(policy, "storage", policy) if policy is not None else Storage()
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError("storage inspection requires an existing directory")
    size = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                size += path.stat().st_size
        except FileNotFoundError:
            # Atomic checkpoint replacement can remove a temporary file while
            # the monitor is walking the active run directory.
            continue
    limit = options.max_run_gb
    return {
        "directory": str(root),
        "bytes": size,
        "limit_bytes": None if limit is None else int(limit * 1024**3),
        "ok": limit is None or size <= limit * 1024**3,
    }


def cleanup(directory, policy=None, *, apply=False):
    from moderntsf.benchmark.infra.policy import Storage

    options = getattr(policy, "storage", policy) if policy is not None else Storage()
    root = Path(directory).resolve()
    if not (root / "manifest.json").is_file():
        raise ValueError("cleanup requires a managed run directory")
    # Never race a controller or live training process, including resource wait.
    with (
        file_lock(root / ".dispatch.lock", blocking=False),
        file_lock(root / ".run.lock", blocking=False),
    ):
        checkpoint = root / "checkpoints"
        candidates = sorted(
            (
                p
                for p in checkpoint.glob("*.pth")
                if not p.is_symlink() and re.fullmatch(r"epoch_\d+\.pth", p.name)
            ),
            key=lambda p: int(re.search(r"\d+", p.name)[0]),
            reverse=True,
        )
        # Best/latest are never candidates. Files named in the latest state are retained.
        protected = set()
        latest = checkpoint / "latest.pth"
        if latest.exists():
            import torch

            saved = torch.load(latest, map_location="cpu", weights_only=False)
            for item in saved.get("top_checkpoints", []):
                for value in item:
                    if isinstance(value, str):
                        protected.add(Path(value).name)
        candidates = [
            p
            for p in candidates[options.keep_epoch_checkpoints :]
            if p.name not in protected
        ]
        result = {
            "apply": apply,
            "files": [str(p) for p in candidates],
            "reclaimed_bytes": sum(p.stat().st_size for p in candidates),
        }
        if apply:
            for path in candidates:
                path.unlink()
        return result

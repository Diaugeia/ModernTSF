"""Read-only accelerator inventory, without experiment or audit dependencies."""

import subprocess


def gpu_inventory() -> list[dict]:
    """Query NVIDIA metadata without importing a training runtime."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 5:
            devices.append(
                dict(zip(("index", "uuid", "name", "free_mb", "total_mb"), fields))
            )
    return devices

"""Shared test setup: fixture data that is generated, not committed."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def smoke_dataset() -> Path:
    """Create ``dataset/smoke/smoke.csv`` (gitignored) once, as CI starts from a clean tree."""
    target = ROOT / "dataset" / "smoke" / "smoke.csv"
    if not target.is_file():
        sys.path.insert(0, str(ROOT / "scripts"))
        import make_smoke_data

        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            make_smoke_data.main()
        finally:
            os.chdir(cwd)
    return target

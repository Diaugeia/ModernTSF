"""Helper for component numerical-regression fixtures.

``assert_reference(name, tensors)`` compares a dict of tensors against
``tests/fixtures/components/<name>.pt``. Set ``TSFLAB_REGEN_COMPONENT_FIXTURES=1``
to (re)write the fixture from the current code; do that only for an intentional,
reviewed numerical change. Store small tensors only (keep each fixture < ~50 KB).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "components"


def assert_reference(name: str, tensors: dict[str, torch.Tensor], *, atol: float = 1e-5, rtol: float = 1e-4) -> None:
    path = FIXTURE_DIR / f"{name}.pt"
    current = {k: v.detach().cpu().clone() for k, v in tensors.items()}
    if os.environ.get("TSFLAB_REGEN_COMPONENT_FIXTURES") == "1":
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(current, path)
        return
    expected = torch.load(path, map_location="cpu", weights_only=True)
    assert set(expected) == set(current), (name, sorted(expected), sorted(current))
    for key, value in current.items():
        ref = expected[key]
        assert ref.shape == value.shape, (name, key, ref.shape, value.shape)
        torch.testing.assert_close(value, ref, atol=atol, rtol=rtol, msg=lambda m: f"{name}.{key}: {m}")

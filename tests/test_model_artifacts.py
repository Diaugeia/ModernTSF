"""Tests for explicit, checksum-verified model artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import BaseModel
import torch.nn as nn

from tsflab.benchmark.model_artifacts import artifact_status, fetch_artifact, require_artifacts
from tsflab.benchmark.registry.models import ModelArtifact, ModelSpec


class _Params(BaseModel):
    width: int = 1


def _spec(source: Path, digest: str) -> ModelSpec:
    artifact = ModelArtifact(
        name="weights",
        url=source.as_uri(),
        revision="commit-123",
        sha256=digest,
        filename="weights.bin",
        required=True,
    )
    return ModelSpec(
        name="Fixture",
        module="tsflab.models.fixture",
        model_class=nn.Identity,
        factory=lambda cfg, params: nn.Identity(),
        params_schema=_Params,
        capabilities=frozenset({"time-series"}),
        artifacts=(artifact,),
    )


def test_fetch_artifact_verifies_and_reuses_cache(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"pinned model artifact")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    spec = _spec(source, digest)
    cache = tmp_path / "cache"

    destination = fetch_artifact(spec, spec.artifacts[0], cache)
    assert destination.read_bytes() == source.read_bytes()
    assert artifact_status(spec, cache)[0]["verified"] is True
    assert require_artifacts(spec, cache)["weights"] == destination
    assert fetch_artifact(spec, spec.artifacts[0], cache) == destination


def test_fetch_artifact_rejects_wrong_checksum(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"unexpected")
    spec = _spec(source, "0" * 64)

    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch_artifact(spec, spec.artifacts[0], tmp_path / "cache")
    assert artifact_status(spec, tmp_path / "cache")[0]["present"] is False
    with pytest.raises(FileNotFoundError, match="tsf model artifacts Fixture"):
        require_artifacts(spec, tmp_path / "cache")


def test_artifact_rejects_unpinned_or_unsafe_fields() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        ModelArtifact("weights", "https://example.com/w", "v1", "bad", "w.bin")
    with pytest.raises(ValueError, match="basename"):
        ModelArtifact("weights", "https://example.com/v1/w", "v1", "0" * 64, "../w")
    with pytest.raises(ValueError, match="pinned revision"):
        ModelArtifact("weights", "https://example.com/w", "v1", "0" * 64, "w")


def test_artifact_factory_receives_validated_params_and_verified_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"foundation fixture")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    received = {}

    def build_with_artifacts(cfg, params, paths):
        received.update({"cfg": cfg, "params": params, "paths": paths})
        return nn.Identity()

    spec = _spec(source, digest)
    spec = ModelSpec(
        **{
            **spec.__dict__,
            "artifact_factory": build_with_artifacts,
        }
    )
    cache = tmp_path / "cache"
    fetched = fetch_artifact(spec, spec.artifacts[0], cache)
    model = spec.build_with_artifacts(
        "config", {"width": 7}, require_artifacts(spec, cache)
    )
    assert isinstance(model, nn.Identity)
    assert received == {
        "cfg": "config",
        "params": {"width": 7},
        "paths": {"weights": fetched},
    }


def test_artifact_factory_requires_a_declared_artifact() -> None:
    with pytest.raises(ValueError, match="without artifacts"):
        ModelSpec(
            name="Fixture",
            module="tsflab.models.fixture",
            model_class=nn.Identity,
            factory=lambda cfg, params: nn.Identity(),
            artifact_factory=lambda cfg, params, paths: nn.Identity(),
            params_schema=_Params,
            capabilities=frozenset({"time-series"}),
        )

"""Strict reader for the declarative model-verification manifest."""

from __future__ import annotations

from pathlib import Path
import tomllib

from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_MANIFEST = Path("verification/models.toml")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ModelVerification(_StrictModel):
    """Per-model executable profile and optional structure-specific test."""

    profile: str = "standard"
    test: str | None = None
    reference_test: str | None = None
    reference_sources: list[str] = Field(default_factory=list)
    structure: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _nonempty_values(self) -> "ModelVerification":
        if not self.profile.strip():
            raise ValueError("verification profile must not be empty")
        if self.test is not None and not self.test.strip():
            raise ValueError("verification test must not be empty")
        if self.reference_test is not None and not self.reference_test.strip():
            raise ValueError("reference comparison test must not be empty")
        if any(not source.strip() for source in self.reference_sources):
            raise ValueError("reference source entries must not be empty")
        if any(not item.strip() for item in self.structure):
            raise ValueError("structure entries must not be empty")
        return self


class VerificationManifest(_StrictModel):
    schema_version: int
    models: dict[str, ModelVerification]

    @model_validator(mode="after")
    def _supported_schema(self) -> "VerificationManifest":
        if self.schema_version != 1:
            raise ValueError(f"unsupported verification manifest schema {self.schema_version}")
        if not self.models:
            raise ValueError("verification manifest contains no models")
        return self


def load_manifest(root: Path, catalog_names: set[str] | None = None) -> VerificationManifest:
    """Load the manifest and optionally require exact catalog coverage."""
    path = root / DEFAULT_MANIFEST
    manifest = VerificationManifest.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    if catalog_names is not None:
        declared = set(manifest.models)
        missing = sorted(catalog_names - declared)
        unknown = sorted(declared - catalog_names)
        problems = []
        if missing:
            problems.append(f"missing models: {', '.join(missing)}")
        if unknown:
            problems.append(f"unknown models: {', '.join(unknown)}")
        if problems:
            raise ValueError("verification manifest/catalog mismatch: " + "; ".join(problems))
    return manifest

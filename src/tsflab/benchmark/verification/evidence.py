"""Canonical, route-free verification evidence and generated index."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tsflab.benchmark.verification_common import ExecutionEnvironment, verification_subject_sha256


SCHEMA_VERSION = 1
DEFAULT_INDEX = Path("verification/index.json")
EVIDENCE_DIRECTORY = Path("verification/evidence")
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PaperReference(_StrictModel):
    title: str
    venue: str
    year: int | None
    url: str


class CodebaseReference(_StrictModel):
    url: str
    revision: str
    license: str


class CheckResult(_StrictModel):
    """One verification result with enough trace to reproduce the decision."""

    status: Literal["passed", "failed", "not-applicable"]
    evidence: list[str] = Field(default_factory=list)
    metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)

    @field_validator("evidence")
    @classmethod
    def _nonempty_entries(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("check evidence entries must not be empty")
        return values

    @model_validator(mode="after")
    def _trace_required(self) -> "CheckResult":
        if self.status != "not-applicable" and not self.evidence:
            raise ValueError("passed/failed checks require evidence")
        if self.status == "not-applicable" and "reason" not in self.metrics:
            raise ValueError("not-applicable checks require a reason metric")
        return self


class VerificationChecks(_StrictModel):
    paper_structure: CheckResult
    equations: CheckResult
    construction: CheckResult
    forward: CheckResult
    backward: CheckResult
    finite_outputs: CheckResult
    active_parameter_gradients: CheckResult
    state_dict_round_trip: CheckResult
    cpu: CheckResult
    batch_size_boundary: CheckResult
    sequence_length_boundary: CheckResult
    input_contract: CheckResult
    reference_comparison: CheckResult



def _is_inference_only(model: str) -> bool:
    """Read the admitted model capability without importing its runtime."""
    from tsflab.benchmark.catalog_metadata import declared_model_fields
    from tsflab.benchmark.registry.models import MODEL_CATALOG
    from tsflab.tsf_core.paths import repository_root

    module = MODEL_CATALOG.refs().get(model)
    if module is None:
        return False
    spec_path = repository_root() / "src" / Path(*module.split(".")).with_suffix(".py")
    capabilities = declared_model_fields(spec_path).get("capabilities", ())
    return "inference-only" in capabilities


class VerificationEvidence(_StrictModel):
    """The only persisted verification record for a model."""

    schema_version: Literal[1]
    model: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
    status: Literal["passed", "failed"]
    verified_at: datetime
    subject_sha256: str = Field(pattern=_SHA256_PATTERN)
    paper: PaperReference
    codebase: CodebaseReference | None
    checks: VerificationChecks
    environment: ExecutionEnvironment
    commands: list[str] = Field(min_length=1)

    @field_validator("verified_at")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("verified_at must include a timezone")
        return value

    @field_validator("commands")
    @classmethod
    def _commands(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("commands must not be empty")
        return values

    @model_validator(mode="after")
    def _status_matches_checks(self) -> "VerificationEvidence":
        allowed_not_applicable = (
            {"backward", "active_parameter_gradients"}
            if _is_inference_only(self.model)
            else set()
        )
        for name, check in self.checks.__dict__.items():
            if name == "reference_comparison":
                continue
            if check.status == "not-applicable" and name not in allowed_not_applicable:
                raise ValueError(f"{name} is a required verification check")
        core_passed = all(
            check.status == "passed"
            or (name in allowed_not_applicable and check.status == "not-applicable")
            for name, check in self.checks.__dict__.items()
            if name != "reference_comparison"
        )
        reference_passed = self.checks.reference_comparison.status != "failed"
        expected = "passed" if core_passed and reference_passed else "failed"
        if self.status != expected:
            raise ValueError("status must agree with all required checks")
        reference_status = self.checks.reference_comparison.status
        if self.codebase is None and reference_status != "not-applicable":
            raise ValueError("models without an official codebase require a not-applicable reference comparison")
        if self.codebase is not None and reference_status == "not-applicable":
            raise ValueError("models with an official codebase require a reference comparison")
        return self


class IndexEntry(_StrictModel):
    evidence: str
    sha256: str = Field(pattern=_SHA256_PATTERN)
    subject_sha256: str = Field(pattern=_SHA256_PATTERN)
    verified_at: datetime
    status: Literal["passed", "failed"]

    @field_validator("evidence")
    @classmethod
    def _safe_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not value.strip():
            raise ValueError("evidence path must be repository-relative")
        return value


class VerificationIndex(_StrictModel):
    schema_version: Literal[1]
    models: dict[str, IndexEntry]


@dataclass(frozen=True)
class VerificationState:
    model: str
    status: Literal["passed", "failed"]
    current: bool
    detail: str | None = None
    evidence: str | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_index(path: Path) -> VerificationIndex:
    return VerificationIndex.model_validate_json(path.read_text(encoding="utf-8"))


def evidence_state(root: Path, name: str, fields: dict[str, object]) -> VerificationState:
    """Return only passed/failed publicly while retaining a stale explanation."""
    index_path = root / DEFAULT_INDEX
    if not index_path.is_file():
        return VerificationState(name, "failed", False, "verification index is missing")
    try:
        index = load_index(index_path)
    except (OSError, ValueError) as exc:
        return VerificationState(name, "failed", False, f"invalid verification index: {exc}")
    entry = index.models.get(name)
    if entry is None:
        return VerificationState(name, "failed", False, "verification evidence is missing")
    path = root / entry.evidence
    if not path.is_file():
        return VerificationState(name, "failed", False, "indexed evidence file is missing", entry.evidence)
    if file_sha256(path) != entry.sha256:
        return VerificationState(name, "failed", False, "evidence digest differs from index", entry.evidence)
    try:
        evidence = VerificationEvidence.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return VerificationState(name, "failed", False, f"invalid evidence: {exc}", entry.evidence)
    if evidence.model != name:
        return VerificationState(name, "failed", False, "evidence model does not match index", entry.evidence)
    from tsflab.tsf_core.paths import is_packaged_root

    # Wheel assets are immutable and intentionally omit checkout-only tests.
    # The index still authenticates the complete evidence document; freshness
    # was established when that evidence was generated in the checkout.
    if is_packaged_root(root):
        consistent = (
            evidence.subject_sha256 == entry.subject_sha256
            and evidence.verified_at == entry.verified_at
            and evidence.status == entry.status
        )
        if not consistent:
            return VerificationState(
                name, "failed", False, "packaged verification metadata disagrees", entry.evidence
            )
        return VerificationState(name, evidence.status, True, evidence=entry.evidence)
    try:
        current_subject = verification_subject_sha256(root, fields)
    except ValueError as exc:
        return VerificationState(name, "failed", False, str(exc), entry.evidence)
    consistent = (
        evidence.subject_sha256 == entry.subject_sha256 == current_subject
        and evidence.verified_at == entry.verified_at
        and evidence.status == entry.status
    )
    if not consistent:
        return VerificationState(name, "failed", False, "verification evidence is stale", entry.evidence)
    return VerificationState(name, evidence.status, True, evidence=entry.evidence)


def rebuild_index(root: Path) -> VerificationIndex:
    """Validate all evidence and atomically generate the compact index."""
    entries: dict[str, IndexEntry] = {}
    for path in sorted((root / EVIDENCE_DIRECTORY).glob("*.json")):
        evidence = VerificationEvidence.model_validate_json(path.read_text(encoding="utf-8"))
        if evidence.model in entries:
            raise ValueError(f"duplicate verification evidence for {evidence.model}")
        relative = path.relative_to(root).as_posix()
        entries[evidence.model] = IndexEntry(
            evidence=relative,
            sha256=file_sha256(path),
            subject_sha256=evidence.subject_sha256,
            verified_at=evidence.verified_at,
            status=evidence.status,
        )
    index = VerificationIndex(schema_version=SCHEMA_VERSION, models=entries)
    target = root / DEFAULT_INDEX
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(index.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return index


def write_evidence(root: Path, evidence: VerificationEvidence) -> Path:
    """Atomically persist one canonical model evidence document."""
    directory = root / EVIDENCE_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{evidence.model}.json"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(evidence.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return target

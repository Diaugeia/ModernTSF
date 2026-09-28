"""Contracts for the single model-verification schema and generated index."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moderntsf.benchmark.verification import (
    VerificationEvidence,
    evidence_state,
    load_manifest,
    rebuild_index,
)
from moderntsf.benchmark.verification.evidence import file_sha256
from moderntsf.benchmark.verification_common import verification_subject_sha256
from moderntsf.benchmark.commands.verification import _load_existing, _materially_changed


def _check(status: str = "passed") -> dict[str, object]:
    payload: dict[str, object] = {"status": status, "evidence": ["tests/test_example.py"]}
    if status == "not-applicable":
        payload = {"status": status, "metrics": {"reason": "no official codebase"}}
    return payload


def _payload(subject: str, *, reference: str = "not-applicable") -> dict[str, object]:
    checks = {
        name: _check()
        for name in (
            "paper_structure",
            "equations",
            "construction",
            "forward",
            "backward",
            "finite_outputs",
            "active_parameter_gradients",
            "state_dict_round_trip",
            "cpu",
            "batch_size_boundary",
            "sequence_length_boundary",
            "input_contract",
        )
    }
    checks["reference_comparison"] = _check(reference)
    return {
        "schema_version": 1,
        "model": "Example",
        "status": "failed" if reference == "failed" else "passed",
        "verified_at": datetime.now(UTC),
        "subject_sha256": subject,
        "paper": {"title": "Example", "venue": "Test", "year": 2026, "url": "https://example.com/paper"},
        "codebase": None,
        "checks": checks,
        "environment": {
            "python": "3.12",
            "framework": "torch 2.6.0",
            "dependencies": {"torch": "2.6.0"},
            "platform": "test",
            "device": "cpu",
            "dtype": "float32",
            "deterministic": {"seed": 1},
        },
        "commands": ["uv run tsf verify model Example"],
    }


class VerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        package = self.root / "src/moderntsf/models/example"
        package.mkdir(parents=True)
        (package / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
        (package / "README.md").write_text("model card\n", encoding="utf-8")
        config = self.root / "configs/models/example.toml"
        config.parent.mkdir(parents=True)
        config.write_text("[model]\n", encoding="utf-8")
        verification = self.root / "verification"
        verification.mkdir()
        (verification / "models.toml").write_text(
            "schema_version = 1\n\n"
            "[models.Example]\n"
            "test = 'tests/test_example.py'\n",
            encoding="utf-8",
        )
        tests = self.root / "tests"
        tests.mkdir()
        (tests / "test_example.py").write_text("VALUE = 1\n", encoding="utf-8")
        self.fields: dict[str, object] = {
            "name": "Example",
            "package": "example",
            "config_path": "configs/models/example.toml",
            "components": (),
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_evidence(self, payload: dict[str, object]) -> Path:
        path = self.root / "verification/evidence/Example.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        evidence = VerificationEvidence.model_validate(payload)
        path.write_text(evidence.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    def test_index_is_generated_from_one_evidence_file_per_model(self) -> None:
        subject = verification_subject_sha256(self.root, self.fields)
        path = self._write_evidence(_payload(subject))
        index = rebuild_index(self.root)
        self.assertEqual(set(index.models), {"Example"})
        self.assertEqual(index.models["Example"].sha256, file_sha256(path))
        state = evidence_state(self.root, "Example", self.fields)
        self.assertEqual((state.status, state.current), ("passed", True))

    def test_missing_invalid_and_stale_evidence_fail_closed(self) -> None:
        missing = evidence_state(self.root, "Example", self.fields)
        self.assertEqual((missing.status, missing.current), ("failed", False))

        subject = verification_subject_sha256(self.root, self.fields)
        path = self._write_evidence(_payload(subject))
        rebuild_index(self.root)
        path.write_text("{}\n", encoding="utf-8")
        invalid = evidence_state(self.root, "Example", self.fields)
        self.assertEqual((invalid.status, invalid.current), ("failed", False))
        self.assertIn("digest", invalid.detail or "")

        self._write_evidence(_payload(subject))
        rebuild_index(self.root)
        (self.root / "src/moderntsf/models/example/model.py").write_text("VALUE = 2\n", encoding="utf-8")
        stale = evidence_state(self.root, "Example", self.fields)
        self.assertEqual((stale.status, stale.current), ("failed", False))
        self.assertIn("stale", stale.detail or "")

    def test_declared_verification_test_is_part_of_the_subject(self) -> None:
        subject = verification_subject_sha256(self.root, self.fields)
        self._write_evidence(_payload(subject))
        rebuild_index(self.root)
        (self.root / "tests/test_example.py").write_text("VALUE = 2\n", encoding="utf-8")
        state = evidence_state(self.root, "Example", self.fields)
        self.assertEqual((state.status, state.current), ("failed", False))
        self.assertIn("stale", state.detail or "")

    def test_packaged_assets_use_authenticated_build_time_evidence(self) -> None:
        subject = verification_subject_sha256(self.root, self.fields)
        self._write_evidence(_payload(subject))
        rebuild_index(self.root)
        (self.root / ".packaged-assets").write_text("immutable\n", encoding="utf-8")
        (self.root / "tests/test_example.py").unlink()
        state = evidence_state(self.root, "Example", self.fields)
        self.assertEqual((state.status, state.current), ("passed", True))

    def test_repeat_verification_does_not_rewrite_timestamp_only(self) -> None:
        first = VerificationEvidence.model_validate(_payload("a" * 64))
        second_payload = _payload("a" * 64)
        second_payload["verified_at"] = datetime.now(UTC)
        second = VerificationEvidence.model_validate(second_payload)
        self.assertFalse(_materially_changed(first, second))
        changed_payload = second.model_dump(mode="python")
        changed_payload["checks"]["forward"] = _check("failed")
        changed_payload["status"] = "failed"
        changed = VerificationEvidence.model_validate(changed_payload)
        self.assertTrue(_materially_changed(first, changed))

        path = self._write_evidence(first.model_dump(mode="python"))
        self.assertEqual(_load_existing(path), first)

    def test_only_reference_comparison_may_be_not_applicable(self) -> None:
        payload = _payload("a" * 64)
        payload["checks"]["forward"] = _check("not-applicable")
        payload["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "required verification check"):
            VerificationEvidence.model_validate(payload)

    def test_inference_only_allows_only_training_checks_to_be_not_applicable(self) -> None:
        payload = _payload("a" * 64)
        payload["checks"]["backward"] = _check("not-applicable")
        payload["checks"]["active_parameter_gradients"] = _check("not-applicable")
        with patch("moderntsf.benchmark.verification.evidence._is_inference_only", return_value=True):
            evidence = VerificationEvidence.model_validate(payload)
            self.assertEqual(evidence.status, "passed")

        payload["checks"]["forward"] = _check("not-applicable")
        with (
            patch("moderntsf.benchmark.verification.evidence._is_inference_only", return_value=True),
            self.assertRaisesRegex(ValueError, "required verification check"),
        ):
            VerificationEvidence.model_validate(payload)

    def test_failed_reference_comparison_fails_the_model(self) -> None:
        payload = _payload("a" * 64, reference="failed")
        payload["codebase"] = {
            "url": "https://example.com/code",
            "revision": "0123456789abcdef",
            "license": "MIT",
        }
        evidence = VerificationEvidence.model_validate(payload)
        self.assertEqual(evidence.status, "failed")

    def test_repository_manifest_exactly_covers_the_catalog(self) -> None:
        from moderntsf.benchmark.catalog_metadata import model_records
        from moderntsf.tsf_core.paths import repository_root

        root = repository_root()
        names = {str(record["name"]) for record in model_records(root)}
        manifest = load_manifest(root, names)
        self.assertEqual(set(manifest.models), names)
        self.assertEqual(len(names), 182)
        self.assertTrue(all(item.test for item in manifest.models.values()))
        saved = json.loads((root / "verification/index.json").read_text(encoding="utf-8"))
        self.assertEqual(set(saved["models"]), names)

        records = {str(record["name"]): record for record in model_records(root)}
        for name, declaration in manifest.models.items():
            evidence = json.loads(
                (root / "verification" / "evidence" / f"{name}.json").read_text(
                    encoding="utf-8"
                )
            )
            codebase = records[name]["codebase"]
            if codebase is not None:
                self.assertIsNotNone(declaration.reference_test, name)
                self.assertEqual(
                    evidence["checks"]["reference_comparison"]["status"],
                    "passed",
                    name,
                )
                prefix = f"{codebase['url']}/blob/{codebase['revision']}/"
                self.assertTrue(
                    all(source.startswith(prefix) for source in declaration.reference_sources),
                    name,
                )


if __name__ == "__main__":
    unittest.main()

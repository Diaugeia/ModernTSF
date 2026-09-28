"""Regression tests for catalog, profiling, and training-layer boundaries."""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from pydantic import ValidationError
from torch import nn

from moderntsf.benchmark.catalog_metadata import model_records
from moderntsf.benchmark.config.schema.evaluation import EvaluationConfig
from moderntsf.benchmark.config.schema.runtime import ExperimentRuntimeConfig
from moderntsf.benchmark.config.schema.training import TrainConfig
from moderntsf.benchmark.registry.losses import LOSS_NAME_MAP
from moderntsf.benchmark.registry.models import MODEL_CATALOG
from moderntsf.benchmark.runner.trainer import _forward_training
from moderntsf.benchmark.utils.record import write_run_record
from moderntsf.data.schemas.datasets.custom import DatasetParameterConfig


ROOT = Path(__file__).resolve().parents[1]


class _FourInputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[object, ...]] = []

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        self.calls.append((x_enc, x_mark_enc, x_dec, x_mark_dec))
        return x_enc[:, -2:, :]


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_model_catalog_listing_does_not_import_model_runtimes(self) -> None:
        script = """
import importlib.abc
import io
import json
import sys
from contextlib import redirect_stdout

class BlockModelRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"torch", "numpy"} or fullname.startswith(("torch.", "numpy.")):
            raise ModuleNotFoundError(f"{fullname} intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockModelRuntime())
from moderntsf.benchmark.cli import main
from moderntsf.benchmark.registry.models import MODEL_CATALOG
assert len(MODEL_CATALOG.names()) == 186
output = io.StringIO()
with redirect_stdout(output):
    assert main(["model", "list", "--json"]) == 0
assert len(json.loads(output.getvalue())) == 186
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_retired_runtime_and_loss_aliases_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ExperimentRuntimeConfig.model_validate({"gpus": [0, 1]})
        self.assertNotIn("l1", LOSS_NAME_MAP)

    def test_experiment_and_dataset_schemas_reject_unknown_options(self) -> None:
        with self.assertRaises(ValidationError):
            EvaluationConfig.model_validate({"profiling": True})
        with self.assertRaises(ValidationError):
            TrainConfig.model_validate({"epochs": 1, "learning_rate": 0.01})
        with self.assertRaises(ValidationError):
            DatasetParameterConfig.model_validate(
                {"target": "OT", "normalise_each_channel": True}
            )

    def test_invalid_run_record_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "records" / "invalid.json"
            with (
                patch(
                    "moderntsf.benchmark.utils.record.build_record_dict",
                    side_effect=ValueError("invalid record"),
                ),
                self.assertRaisesRegex(ValueError, "invalid record"),
            ):
                write_run_record(str(target))
            self.assertFalse(target.exists())

    def test_catalog_metadata_uses_registration_as_admission_boundary(self) -> None:
        records = model_records(ROOT, refs={"Linear": "moderntsf.models.linear.spec"})
        self.assertEqual([record["name"] for record in records], ["Linear"])

    def test_profiler_uses_canonical_four_input_call_and_restores_mode(self) -> None:
        profile = importlib.import_module("moderntsf.benchmark.evaluation.profile")
        model = _FourInputModel().eval()
        loader = [
            (
                torch.randn(2, 4, 3),
                torch.randn(2, 2, 3),
                torch.randn(2, 4, 2),
                torch.randn(2, 2, 2),
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "profile.txt"
            with (
                patch.object(profile, "_try_torchinfo_summary", return_value="summary"),
                patch.object(profile, "_try_flops", return_value="flops"),
                patch.object(profile, "_latency_benchmark", return_value=["latency"]),
            ):
                profile.profile_model(
                    model, loader, torch.device("cpu"), 0, 2, str(target)
                )
            self.assertTrue(target.is_file())
            self.assertIn("summary", target.read_text(encoding="utf-8"))
        self.assertFalse(model.training)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(model.calls[0]), 4)

    def test_declared_training_objective_replaces_configured_criterion(self) -> None:
        spec = MODEL_CATALOG.get("DistDF")
        model = spec.model_class(8, 3, 2)
        batch_x = torch.randn(4, 8, 2)
        batch_y = torch.randn(4, 3, 2)

        def forbidden_criterion(*_args):
            raise AssertionError("configured criterion must not replace paper objective")

        outputs, loss = _forward_training(
            model,
            spec.training_objective,
            batch_x,
            None,
            torch.zeros_like(batch_y),
            None,
            batch_y,
            3,
            "M",
            forbidden_criterion,
        )
        self.assertEqual(tuple(outputs.shape), tuple(batch_y.shape))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_inference_only_capability_rejects_training_hooks(self) -> None:
        spec = MODEL_CATALOG.get("Linear")
        with self.assertRaisesRegex(ValueError, "pretraining-stage"):
            replace(
                spec,
                capabilities=spec.capabilities | {"inference-only", "pretraining-stage"},
            )
        with self.assertRaisesRegex(ValueError, "training objective"):
            replace(
                spec,
                capabilities=spec.capabilities | {"inference-only"},
                training_objective=lambda *_args: None,
            )


if __name__ == "__main__":
    unittest.main()

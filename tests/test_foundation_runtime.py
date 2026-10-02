"""Tests for the optional, offline foundation-model runtime boundary."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tsflab.models._foundation import (
    ChronosRuntime,
    FoundationForecast,
    FoundationModel,
    FoundationSource,
    MoiraiRuntime,
    TimesFMRuntime,
)


def _source(name: str = "Example") -> FoundationSource:
    return FoundationSource(
        name=name,
        package="official-package==1.0",
        codebase="https://example.com/official",
        model_id="owner/model",
        revision="0123456789abcdef",
        license="Apache-2.0",
    )


class _Runtime:
    source = _source()

    def predict(self, context, prediction_length, quantile_levels):
        mean = context[:, -1:].repeat(1, prediction_length)
        quantiles = None
        if quantile_levels:
            quantiles = torch.stack(
                [mean + level for level in quantile_levels], dim=-1
            )
        return FoundationForecast(mean, quantiles)


class FoundationRuntimeTests(unittest.TestCase):
    def test_canonical_wrapper_restores_channel_axes(self) -> None:
        values = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        point = FoundationModel(_Runtime(), prediction_length=2)
        self.assertEqual(tuple(point(values).shape), (2, 2, 3))

        quantile = FoundationModel(
            _Runtime(), prediction_length=2, quantile_levels=(0.1, 0.5, 0.9)
        )
        output = quantile(values)
        self.assertEqual(tuple(output.shape), (2, 2, 3, 3))
        self.assertTrue(torch.all(output[..., 0] < output[..., 1]))

    def test_source_rejects_incomplete_or_non_https_facts(self) -> None:
        with self.assertRaises(ValueError):
            FoundationSource("", "pkg", "https://example.com", "id", "rev", "MIT")
        with self.assertRaises(ValueError):
            FoundationSource("x", "pkg", "http://example.com", "id", "rev", "MIT")

    def test_chronos_loader_is_pinned_and_offline(self) -> None:
        calls = {}

        class Pipeline:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls.update(path=path, **kwargs)
                return cls()

            def predict_quantiles(self, *, inputs, prediction_length, quantile_levels):
                mean = torch.zeros(inputs.shape[0], prediction_length)
                quantiles = torch.zeros(
                    inputs.shape[0], prediction_length, len(quantile_levels)
                )
                return quantiles, mean

        with tempfile.TemporaryDirectory() as directory:
            runtime = ChronosRuntime.from_local(
                _source("Chronos"), directory, loader=Pipeline
            )
        result = runtime.predict(torch.ones(2, 8), 3, (0.1, 0.9))
        self.assertTrue(calls["local_files_only"])
        self.assertEqual(calls["revision"], "0123456789abcdef")
        self.assertEqual(tuple(result.quantiles.shape), (2, 3, 2))

    def test_timesfm_loader_compiles_and_selects_requested_deciles(self) -> None:
        calls = {}

        class Model:
            def compile(self, config):
                calls["config"] = config

            def forecast(self, *, horizon, inputs):
                point = np.zeros((len(inputs), horizon), dtype=np.float32)
                quantiles = np.zeros((len(inputs), horizon, 10), dtype=np.float32)
                quantiles[..., 1] = 0.1
                quantiles[..., 9] = 0.9
                return point, quantiles

        class Loader:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls.update(path=path, **kwargs)
                return Model()

        def config_factory(**kwargs):
            return kwargs

        with tempfile.TemporaryDirectory() as directory:
            runtime = TimesFMRuntime.from_local(
                _source("TimesFM"),
                directory,
                max_context=32,
                max_horizon=8,
                loader=Loader,
                config_factory=config_factory,
            )
        result = runtime.predict(torch.ones(2, 16), 4, (0.1, 0.9))
        self.assertTrue(calls["local_files_only"])
        self.assertEqual(calls["config"]["max_horizon"], 8)
        self.assertTrue(torch.allclose(result.quantiles[..., 0], torch.tensor(0.1)))
        self.assertTrue(torch.allclose(result.quantiles[..., 1], torch.tensor(0.9)))

    def test_moirai_adapter_uses_official_quantile_output(self) -> None:
        class Module:
            quantile_levels = (0.1, 0.5, 0.9)

        class HParams:
            prediction_length = 2

        class Model:
            module = Module()
            hparams = HParams()

            def predict(self, inputs):
                batch = len(inputs)
                values = np.zeros((batch, 3, 2, 1), dtype=np.float32)
                values[:, 1, :, :] = 0.5
                return values

        runtime = MoiraiRuntime(_source("Moirai"), Model())
        result = runtime.predict(torch.ones(2, 8), 2, (0.1, 0.9))
        self.assertEqual(tuple(result.mean.shape), (2, 2))
        self.assertEqual(tuple(result.quantiles.shape), (2, 2, 2))
        self.assertTrue(torch.allclose(result.mean, torch.tensor(0.5)))

    def test_official_loaders_require_an_existing_local_path(self) -> None:
        class Loader:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                raise AssertionError("loader must not run for a missing path")

        missing = Path(tempfile.gettempdir()) / "tsflab-missing-foundation"
        with self.assertRaises(FileNotFoundError):
            ChronosRuntime.from_local(_source("Chronos"), missing, loader=Loader)


if __name__ == "__main__":
    unittest.main()

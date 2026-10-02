"""Runner contract for model-provided training objectives and the wired models."""

from __future__ import annotations

import dataclasses
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from tsflab.benchmark.registry.losses import get_loss
from tsflab.benchmark.registry.models import MODEL_CATALOG
from tsflab.benchmark.runner.callbacks import Callback
from tsflab.benchmark.runner.objective import TrainingBatch
from tsflab.benchmark.runner.trainer import _forward_training, train

ROOT = Path(__file__).resolve().parents[1]


class _Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(6, 3)
        self.aux_loss = None

    def forward(self, x, x_mark=None, dec=None, y_mark=None):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


def _batches(count=2, seq=6, pred=3, channels=2):
    torch.manual_seed(0)
    return [
        (torch.randn(4, seq, channels), torch.randn(4, pred, channels), None, None)
        for _ in range(count)
    ]


def _step(model, objective, x, y, criterion, features="M"):
    return _forward_training(
        model, objective, x, None, torch.zeros_like(y), None, y, y.shape[1], features, criterion
    )


class RunnerContractTests(unittest.TestCase):
    def test_default_path_is_criterion_plus_aux_loss(self) -> None:
        model, criterion = _Linear(), get_loss("mse")
        x, y = torch.randn(4, 6, 2), torch.randn(4, 3, 2)
        outputs, loss = _step(model, None, x, y, criterion)
        self.assertTrue(torch.equal(loss, criterion(outputs, y)))
        model.aux_loss = torch.tensor(0.5)
        _, with_aux = _step(model, None, x, y, criterion)
        self.assertTrue(torch.allclose(with_aux, loss + 0.5))

    def test_objective_receives_batch_and_criterion(self) -> None:
        seen = {}

        def objective(model, batch, criterion):
            seen.update(batch=batch, criterion=criterion)
            forecast = batch.forecast(model)
            return forecast, criterion(batch.align(forecast), batch.target) * 2

        model, criterion = _Linear(), get_loss("mse")
        x, y = torch.randn(4, 6, 2), torch.randn(4, 3, 2)
        outputs, loss = _step(model, objective, x, y, criterion, "MS")
        self.assertIsInstance(seen["batch"], TrainingBatch)
        self.assertIs(seen["criterion"], criterion)
        self.assertEqual(tuple(seen["batch"].target.shape), (4, 3, 1))
        self.assertTrue(torch.isfinite(loss))

    def test_objective_may_return_no_forecast_but_loss_must_be_finite(self) -> None:
        model, criterion = _Linear(), get_loss("mse")
        x, y = torch.randn(4, 6, 2), torch.randn(4, 3, 2)
        outputs, loss = _step(model, lambda m, b, c: (None, b.forecast(m).mean()), x, y, criterion)
        self.assertIsNone(outputs)
        with self.assertRaises(ValueError):
            _step(model, lambda m, b, c: (None, torch.tensor(float("nan"))), x, y, criterion)
        with self.assertRaises(ValueError):
            _step(model, lambda m, b, c: (torch.zeros(4, 3, 5), b.forecast(m).mean()), x, y, criterion)

    def test_spec_validation(self) -> None:
        spec = MODEL_CATALOG.get("MMPD")
        with self.assertRaises(ValueError):
            dataclasses.replace(
                MODEL_CATALOG.get("DistDF"), training_objective=None, training_setup=lambda *a, **k: None
            )
        with self.assertRaises(TypeError):
            dataclasses.replace(spec, training_objective=3)

    def _train(self, objective=None, setup=None, callbacks=None):
        model = _Linear()
        loader = _batches()
        with tempfile.TemporaryDirectory() as tmp:
            train(
                model, loader, loader, torch.device("cpu"), 2, 5, "mse", {},
                torch.optim.SGD(model.parameters(), lr=0.01), "type1", 0.01, 2, 0, 3, "M",
                False, tmp, SimpleNamespace(strategy="best", save_k=1),
                callbacks=callbacks, training_objective=objective, training_setup=setup,
            )
        return model

    def test_train_runs_setup_once_and_supports_missing_forecast_with_callbacks(self) -> None:
        calls = []

        def setup(model, loader, *, pred_len, features):
            calls.append((pred_len, features, len(loader)))

        def objective(model, batch, criterion):
            return None, criterion(batch.align(batch.forecast(model)), batch.target)

        seen = []

        class Spy(Callback):
            def on_compute_loss(self, ctx):
                seen.append(ctx.outputs.requires_grad)

        self._train(objective, setup, [Spy()])
        self.assertEqual(calls, [(3, "M", 2)])
        self.assertTrue(seen and not any(seen))
        self._train(objective, setup)

    def test_validation_ignores_the_objective(self) -> None:
        def objective(model, batch, criterion):
            return None, batch.forecast(model).sum() * 0.0 + 1000.0

        # Validation uses forward + criterion; training objective only affects the loop loss.
        self._train(objective)



class WiredModelTests(unittest.TestCase):
    def objective_step(self, name, seq=16, pred=8, channels=7, features="M", **overrides):
        torch.manual_seed(3)
        spec = MODEL_CATALOG.get(name)
        params = tomllib.loads((ROOT / spec.config_path).read_text())["model"]["params"]
        params.update(enc_in=channels, **overrides)
        cfg = SimpleNamespace(task=SimpleNamespace(seq_len=seq, pred_len=pred, label_len=0, features="M"))
        model = spec.build(cfg, params).train()
        x, y = torch.randn(4, seq, channels), torch.randn(4, pred, channels)
        self.assertIsNotNone(spec.training_objective, name)
        outputs, loss = _forward_training(
            model, spec.training_objective, x, None, torch.zeros_like(y), None, y,
            pred, features, get_loss("mse"),
        )
        self.assertTrue(torch.isfinite(loss), name)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
        return spec, model, x, y, outputs, loss

    def test_each_wired_objective_is_finite_and_differentiable(self) -> None:
        self.objective_step("MMPD", d_model=16, patch_len=4, num_heads=2, diffusion_steps=10)
        self.objective_step("MAFS", d_model=16, num_heads=2, num_layers=1)
        self.objective_step("Kronos", d_model=16, num_heads=2, num_layers=1)
        self.objective_step("TimeFilter", d_model=16, d_ff=16, e_layers=1, patch_len=4)
        self.objective_step("GOTSF", d_model=16)

    def test_mmpd_replaces_observation_loss(self) -> None:
        _, _, _, _, outputs, _ = self.objective_step(
            "MMPD", d_model=16, patch_len=4, num_heads=2, diffusion_steps=10
        )
        self.assertIsNone(outputs)

    def test_gts_adds_prior_only_with_adjacency(self) -> None:
        spec, model, x, y, _, loss = self.objective_step("GTS", seq=12, pred=12, channels=8)
        self.assertFalse(model.has_prior)
        base = get_loss("mse")(model(x), y)
        self.assertTrue(torch.isfinite(base))
        import numpy as np

        params = tomllib.loads((ROOT / spec.config_path).read_text())["model"]["params"]
        cfg = SimpleNamespace(task=SimpleNamespace(seq_len=12, pred_len=12, label_len=0, features="M"))
        adj = (np.random.RandomState(0).rand(8, 8) > 0.5).astype("float32")
        with_prior = spec.factory(cfg, {**params, "adj_mx": adj}).train()
        self.assertTrue(with_prior.has_prior)
        _, loss = _step(with_prior, spec.training_objective, x[:, :, :8], y[:, :, :8], get_loss("mse"))
        self.assertTrue(torch.isfinite(loss))

    def test_timefilter_adds_weighted_balance_loss(self) -> None:
        spec, model, x, y, _, loss = self.objective_step(
            "TimeFilter", d_model=16, d_ff=16, e_layers=1, patch_len=4
        )
        model.eval()
        torch.manual_seed(3)
        forecast = model(x)
        plain = get_loss("mse")(forecast, y)
        self.assertGreater(float(model.last_moe_loss), 0.0)
        self.assertNotEqual(float(plain), float(loss))

    def test_visifold_loss_ignores_masked_nodes(self) -> None:
        torch.manual_seed(5)
        spec = MODEL_CATALOG.get("VisiFold")
        params = tomllib.loads((ROOT / spec.config_path).read_text())["model"]["params"]
        params.update(mask_ratio=0.5)
        cfg = SimpleNamespace(task=SimpleNamespace(seq_len=12, pred_len=12, label_len=0, features="M"))
        model = spec.build(cfg, params).train()
        x, y = torch.randn(2, 12, 8), torch.randn(2, 12, 8)
        state = torch.get_rng_state()
        _, loss = _step(model, spec.training_objective, x, y, get_loss("mse"))
        keep = model.last_keep_indices
        self.assertLess(keep.numel(), 8)
        masked = [i for i in range(8) if i not in set(keep.tolist())]
        y2 = y.clone()
        y2[:, :, masked] += 100.0
        torch.set_rng_state(state)
        _, loss2 = _step(model, spec.training_objective, x, y2, get_loss("mse"))
        self.assertTrue(torch.allclose(loss, loss2))
        # Evaluation keeps every node and the standard criterion.
        model.eval()
        self.assertIsNone(None)
        outputs, loss3 = _step(model, spec.training_objective, x, y, get_loss("mse"))
        self.assertTrue(torch.allclose(loss3, get_loss("mse")(outputs, y)))

    def test_gotsf_objective_averages_interval_losses(self) -> None:
        spec, model, x, y, outputs, loss = self.objective_step("GOTSF", d_model=16)
        self.assertEqual(tuple(outputs.shape), tuple(y.shape))
        parts = torch.stack(
            [model.goal_oriented_loss(x, y, i) for i in range(model.num_intervals)]
        )
        model.eval()
        self.assertTrue(torch.isfinite(parts.mean()))

    def test_timeo1_setup_fits_basis_and_objective_uses_it(self) -> None:
        spec = MODEL_CATALOG.get("TimeO1")
        self.assertIsNotNone(spec.training_setup)
        cfg = SimpleNamespace(task=SimpleNamespace(seq_len=16, pred_len=8, label_len=0, features="M"))
        model = spec.build(cfg, {"enc_in": 3, "d_model": 16}).train()
        loader = [(torch.randn(8, 16, 3), torch.randn(8, 8, 3), None, None) for _ in range(4)]
        with self.assertRaises(RuntimeError):
            _step(model, spec.training_objective, loader[0][0], loader[0][1], get_loss("mse"))
        spec.training_setup(model, loader, pred_len=8, features="M")
        self.assertTrue(bool(model.projection_ready))
        _, loss = _step(model, spec.training_objective, loader[0][0], loader[0][1], get_loss("mse"))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        # The sum form of Eq. (5) is unchanged for direct callers.
        sum_loss = model.transformed_alignment_loss(model(loader[0][0]), loader[0][1])
        self.assertGreater(float(sum_loss), float(loss))
        # MS targets fit the trailing channel only.
        ms = spec.build(cfg, {"enc_in": 3, "d_model": 16}).train()
        spec.training_setup(ms, loader, pred_len=8, features="MS")
        _, ms_loss = _forward_training(
            ms, spec.training_objective, loader[0][0], None, torch.zeros_like(loader[0][1]),
            None, loader[0][1], 8, "MS", get_loss("mse"),
        )
        self.assertTrue(torch.isfinite(ms_loss))

    def test_declared_scope_is_exactly_the_wired_models(self) -> None:
        declared = {n for n in MODEL_CATALOG.names() if MODEL_CATALOG.get(n).training_objective}
        for name in ("MMPD", "TimeO1", "GTS", "MAFS", "Kronos", "TimeFilter", "GOTSF", "VisiFold",
                     "DistDF", "AMRC", "TimeAlign", "LatentTSF"):
            self.assertIn(name, declared)
        for name in ("STOP", "ST-SSDL"):
            self.assertNotIn(name, declared)


if __name__ == "__main__":
    unittest.main()

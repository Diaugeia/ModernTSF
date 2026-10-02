"""Paper-structure and strict runtime checks for six final implementations."""

from __future__ import annotations

import copy
import unittest

import torch

from tsflab.models.glocalib.model import Model as GlocalIB
from tsflab.models.pattn.model import Model as PAttn
from tsflab.models.phat.model import Model as PHAT, _distance_masks
from tsflab.models.quantile_dlinear.model import Model as QuantileDLinear
from tsflab.models.quantile_patchtst.model import Model as QuantilePatchTST
from tsflab.models.tide.model import Model as TiDE


def make_model(name: str, length: int = 8, horizon: int = 3, channels: int = 2):
    if name == "GlocalIB":
        return GlocalIB(length, horizon, channels, d_model=8, mask_ratio=0.25, kl_weight=0.02)
    if name == "PAttn":
        return PAttn(length, horizon, d_model=8, n_heads=2, patch_len=min(4, length), stride=1, dropout=0.0)
    if name == "PHAT":
        return PHAT(length, horizon, channels, d_model=8, n_heads=2, d_layers=1, attn_dropout=0.0, ffn_dropout=0.0)
    if name == "QuantileDLinear":
        return QuantileDLinear(length, horizon, channels, kernel_size=3, quantile_levels=[0.1, 0.5, 0.9])
    if name == "QuantilePatchTST":
        return QuantilePatchTST(length, horizon, channels, patch_len=min(4, length), stride=1, e_layers=1, d_model=8, n_heads=2, d_ff=16, quantile_levels=[0.1, 0.5, 0.9])
    if name == "TiDE":
        return TiDE(length, horizon, 8, 2, 2, 16, 4, 6, 0.0, True, 2)
    raise KeyError(name)


def call(name: str, model, x: torch.Tensor, historical=None, future=None):
    if name == "TiDE":
        return model(x, historical, None, future)
    return model(x, historical, torch.eye(x.shape[-1]), future)


class RuntimeContractTests(unittest.TestCase):
    names = ("GlocalIB", "PAttn", "PHAT", "QuantileDLinear", "QuantilePatchTST", "TiDE")

    def test_forward_backward_active_gradients_and_round_trip(self):
        torch.manual_seed(260827)
        for name in self.names:
            with self.subTest(model=name):
                model = make_model(name).cpu().train()
                x = torch.randn(2, 8, 2, requires_grad=True)
                historical = torch.randn(2, 8, 6)
                future = torch.randn(2, 3, 6)
                output = call(name, model, x, historical, future)
                expected = (2, 3, 2, 3) if name.startswith("Quantile") else (2, 3, 2)
                self.assertEqual(tuple(output.shape), expected)
                self.assertTrue(torch.isfinite(output).all())
                loss = output.square().mean()
                if getattr(model, "aux_loss", None) is not None:
                    loss = loss + model.aux_loss
                loss.backward()
                self.assertIsNotNone(x.grad)
                self.assertGreater(float(x.grad.abs().max()), 0.0)
                for parameter_name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    self.assertIsNotNone(parameter.grad, parameter_name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)
                    self.assertGreater(float(parameter.grad.abs().max()), 0.0, parameter_name)

                model.eval()
                expected_output = call(name, model, x.detach(), historical, future)
                clone = make_model(name).cpu().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()))
                torch.testing.assert_close(call(name, clone, x.detach(), historical, future), expected_output)

    def test_batch_and_sequence_boundaries(self):
        for name in self.names:
            with self.subTest(model=name):
                model = make_model(name, length=4, horizon=1).eval()
                output = call(name, model, torch.randn(1, 4, 2), torch.randn(1, 4, 6), torch.randn(1, 1, 6))
                self.assertEqual(output.shape[0], 1)
                with self.assertRaises((ValueError, RuntimeError)):
                    call(name, model, torch.randn(1, 3, 2), torch.randn(1, 3, 6), torch.randn(1, 1, 6))


class PaperEquationTests(unittest.TestCase):
    def test_glocal_bottleneck_and_alignment_are_training_only(self):
        model = make_model("GlocalIB")
        model.train()
        model(torch.randn(2, 8, 2))
        self.assertIsNotNone(model.aux_loss)
        self.assertGreaterEqual(float(model.aux_loss.detach()), 0.0)
        self.assertTrue(hasattr(model.encoder, "mean"))
        self.assertTrue(hasattr(model.encoder, "log_variance"))
        model.eval()
        model(torch.randn(2, 8, 2))
        self.assertIsNone(model.aux_loss)

    def test_pattn_is_the_no_position_no_ffn_single_attention_variant(self):
        model = make_model("PAttn")
        self.assertIsInstance(model.attention, torch.nn.MultiheadAttention)
        self.assertFalse(hasattr(model, "position_embedding"))
        self.assertFalse(hasattr(model, "feed_forward"))
        self.assertEqual(len([module for module in model.modules() if isinstance(module, torch.nn.MultiheadAttention)]), 1)

    def test_phat_periodic_distance_sets_and_bucket_path(self):
        positive, negative = _distance_masks(4, torch.device("cpu"))
        self.assertEqual(tuple(positive.shape), (4, 4, 4))
        self.assertEqual(tuple(negative.shape), (4, 4, 4))
        # For query phase 0 and key phase 2, nearer phases enter the positive set.
        self.assertEqual(positive[0, 2].tolist(), [1.0, 1.0, 0.0, 1.0])
        model = make_model("PHAT").eval()
        self.assertEqual(tuple(model._bucket_path(torch.randn(2, 8), 4).shape), (2, 8))

    def test_quantile_axis_monotonicity_and_validation(self):
        for name in ("QuantileDLinear", "QuantilePatchTST"):
            with self.subTest(model=name):
                output = make_model(name).eval()(torch.randn(2, 8, 2))
                self.assertEqual(output.shape[-1], 3)
                self.assertTrue((output[..., 1:] >= output[..., :-1]).all())
        with self.assertRaises(ValueError):
            QuantileDLinear(8, 3, 2, quantile_levels=[0.5, 0.1])
        with self.assertRaises(ValueError):
            QuantilePatchTST(8, 3, 2, quantile_levels=[])

    def test_tide_future_covariate_highway_and_distinct_blocks(self):
        torch.manual_seed(7)
        model = make_model("TiDE").eval()
        x = torch.randn(2, 8, 2)
        historical = torch.randn(2, 8, 6)
        future = torch.randn(2, 3, 6)
        first = model(x, historical, None, future)
        second = model(x, historical, None, future + 1.0)
        self.assertGreater(float((first - second).abs().max()), 0.0)
        blocks = [*model.encoder_blocks, *model.decoder_blocks]
        self.assertEqual(len({id(block) for block in blocks}), len(blocks))


if __name__ == "__main__":
    unittest.main()

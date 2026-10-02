"""Paper-structure and runtime tests for the final eight implementations."""
from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from tsflab.models.pathformer.model import Model as Pathformer
from tsflab.models.stpgnn.model import Model as STPGNN
from tsflab.models.timeemb.model import Model as TimeEmb
from tsflab.models.timefilter.model import Model as TimeFilter
from tsflab.models.timeperceiver.model import Model as TimePerceiver
from tsflab.models.timexer.model import Model as TimeXer
from tsflab.models.umixer.model import Model as UMixer
from tsflab.models.wpmixer.model import Model as WPMixer


def marks(batch=2, steps=12, offset=0):
    rows = [[2026, 8, 1+i//24, 4, (i+offset)%24, 0] for i in range(steps)]
    return torch.tensor([rows]*batch, dtype=torch.float32)


def graph(nodes=4):
    adjacency = np.eye(nodes, dtype=np.float32)
    for index in range(nodes-1):
        adjacency[index, index+1] = 1
    return adjacency


def factories():
    return {
        "Pathformer": lambda: Pathformer(12, 3, 4, layer_nums=1, k=2, num_experts=2, patch_size_list=[3, 4], d_model=8, d_ff=16, n_heads=2, dropout=0),
        "STPGNN": lambda: STPGNN(12, 3, 4, graph(), dropout=0, topk=2, residual_channels=8, end_channels=16, kernel_size=2, blocks=1, layers=1, dims=4),
        "TimeEmb": lambda: TimeEmb(12, 3, 4, d_model=8, use_revin=True, use_hour_index=True, use_day_index=True, scale=.02, hour_length=24, day_length=7),
        "TimeFilter": lambda: TimeFilter(12, 3, 4, d_model=8, d_ff=16, e_layers=1, patch_len=3, dropout=0, top_p=.5, num_experts=2),
        "TimePerceiver": lambda: TimePerceiver(12, 3, 4, d_model=8, n_heads=2, patch_len=3, dropout=0, num_latents=3, latent_dim=8, latent_d_ff=16, num_latent_blocks=1),
        "TimeXer": lambda: TimeXer(12, 3, 4, d_model=8, n_heads=2, e_layers=1, d_ff=16, patch_len=3, dropout=0),
        "UMixer": lambda: UMixer(12, 3, 4, d_model=8, e_layers=2, patch_len=3, stride=3, dropout=0),
        "WPMixer": lambda: WPMixer(12, 3, 4, d_model=8, dropout=0, tfactor=2, dfactor=2, wavelet="db2", level=2, patch_len=3, stride=2),
    }


class PaperStructureTests(unittest.TestCase):
    def test_pathformer_routes_multiple_dual_attention_scales(self):
        model = factories()["Pathformer"]().eval()
        model(torch.randn(2, 12, 4))
        layer = model.layers[0]
        self.assertEqual(layer.patch_sizes, (3, 4))
        self.assertIsNotNone(layer.last_route)
        torch.testing.assert_close(layer.last_route.sum(-1), torch.ones(2, 4))
        self.assertTrue(all(hasattr(expert, "local_attention") and hasattr(expert, "global_attention") for expert in layer.experts))

    def test_stpgnn_pivotal_and_parallel_paths(self):
        model = factories()["STPGNN"]().eval()
        model(torch.randn(2, 12, 4), marks())
        self.assertEqual(model.identifier.last_indices.shape, (2,))
        layer = model.st_layers[0]
        self.assertTrue(hasattr(layer, "pivotal"))
        self.assertTrue(hasattr(layer, "graph_projection"))
        self.assertTrue(hasattr(layer, "temporal"))

    def test_timeemb_disentangles_static_and_dynamic_spectra(self):
        model = factories()["TimeEmb"]().eval()
        model(torch.randn(2, 12, 4), marks(), x_mark_dec=marks(2, 3, 4))
        self.assertEqual(model.last_static_spectrum.shape, (2, 4, 7))
        self.assertTrue(hasattr(model.dynamic_filter, "conditioner"))

    def test_timefilter_filters_channel_patch_graph_with_moe(self):
        model = factories()["TimeFilter"]().eval()
        model(torch.randn(2, 12, 4))
        layer = model.filters[0]
        self.assertEqual(layer.last_adjacency.shape, (2, 16, 16))
        self.assertEqual(layer.last_routes.shape, (2, 16, 2))
        torch.testing.assert_close(layer.last_adjacency.sum(-1), torch.ones(2, 16))
        self.assertTrue(torch.isfinite(model.last_moe_loss))

    def test_timeperceiver_latent_bottleneck_and_target_queries(self):
        model = factories()["TimePerceiver"]().eval()
        model(torch.randn(2, 12, 4), marks(), x_mark_dec=marks(2, 3, 3))
        self.assertEqual(model.last_encoder_attention.shape[-2:], (3, 16))
        self.assertEqual(model.last_decoder_attention.shape[-2:], (3, 3))

    def test_timexer_global_bridge_cross_attends_exogenous_tokens(self):
        model = factories()["TimeXer"]().eval()
        model(torch.randn(2, 12, 4), marks())
        attention = model.layers[0].last_cross_attention
        self.assertEqual(attention.shape, (8, 1, 5))
        torch.testing.assert_close(attention.sum(-1), torch.ones(8, 1))

    def test_umixer_keeps_axes_and_stationarity_correction_distinct(self):
        model = factories()["UMixer"]().eval()
        model(torch.randn(2, 12, 4))
        self.assertEqual(model.down_mixers[0].patch_mlp[0].in_features, 4)
        self.assertEqual(model.down_mixers[0].feature_mlp[0].in_features, 8)
        self.assertIsNone(model.down_mixers[0].patch_mlp[3].bias)
        self.assertIsNotNone(model.correction.last_factor)

    def test_wpmixer_has_all_wavelet_resolutions_and_axis_mixers(self):
        model = factories()["WPMixer"]().eval()
        resolutions = model.wavelet(torch.randn(2, 4, 12))
        self.assertEqual(len(resolutions), 3)
        self.assertEqual(len(model.branches), 3)
        model(torch.randn(2, 12, 4))
        torch.testing.assert_close(model.last_resolution_weights.sum(), torch.tensor(1.0))


class RuntimeContractTests(unittest.TestCase):
    def test_forward_backward_active_parameters_and_round_trip(self):
        torch.manual_seed(260827)
        x = torch.randn(2, 12, 4)
        for name, factory in factories().items():
            with self.subTest(model=name):
                model = factory().cpu().eval()
                value = x.clone().requires_grad_(True)
                output = model(value, marks(), x_mark_dec=marks(2, 3, 5))
                self.assertEqual(output.shape, (2, 3, 4))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(value.grad)
                self.assertGreater(value.grad.abs().max().item(), 0)
                for parameter_name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, parameter_name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)
                    if name == "WPMixer" and parameter_name.endswith(".token_mixer.3.bias"):
                        # This per-patch offset is shared across features and
                        # removed by norm1; residual gradients are roundoff.
                        self.assertLessEqual(parameter.grad.abs().max().item(), 1e-6, parameter_name)
                    else:
                        self.assertGreater(parameter.grad.abs().max().item(), 0, parameter_name)
                clone = factory().cpu().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
                torch.testing.assert_close(clone(x, marks(), x_mark_dec=marks(2, 3, 5)), model(x, marks(), x_mark_dec=marks(2, 3, 5)))
                self.assertEqual(model(x[:1], marks(1), x_mark_dec=marks(1, 3)).shape, (1, 3, 4))
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 11, 4))

    def test_wpmixer_token_bias_is_removed_by_layer_norm(self):
        torch.manual_seed(91)
        model = factories()["WPMixer"]().double().eval()
        x = torch.randn(2, 12, 4, dtype=torch.float64)
        baseline = model(x)
        baseline.square().mean().backward()
        for branch in model.branches:
            self.assertLessEqual(branch.token_mixer[3].bias.grad.abs().max().item(), 1e-12)
        with torch.no_grad():
            for branch in model.branches:
                branch.token_mixer[3].bias.add_(torch.randn_like(branch.token_mixer[3].bias) * 7)
        torch.testing.assert_close(model(x), baseline, rtol=1e-12, atol=1e-12)

    def test_marks_and_adjacency_contracts_are_active(self):
        x = torch.randn(2, 12, 4)
        for name in ("TimeEmb", "TimePerceiver", "TimeXer", "STPGNN"):
            model = factories()[name]().eval()
            first = model(x, marks(offset=0), x_mark_dec=marks(2, 3, 0))
            second = model(x, marks(offset=7), x_mark_dec=marks(2, 3, 7))
            self.assertGreater((first-second).abs().max().item(), 0, name)
        torch.manual_seed(91)
        identity = STPGNN(12, 3, 4, np.eye(4), dropout=0, topk=2, residual_channels=8, end_channels=16, blocks=1, layers=1, dims=4).eval()
        torch.manual_seed(91)
        connected = factories()["STPGNN"]().eval()
        self.assertGreater((identity(x)-connected(x)).abs().max().item(), 0)


if __name__ == "__main__":
    unittest.main()

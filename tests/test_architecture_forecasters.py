"""Paper-structure and runtime tests for architecture batch C."""
from __future__ import annotations

import copy
import unittest

import torch
from pydantic import ValidationError

from moderntsf.models.dtaf.model import FrequencyWaveModeling, TemporalStabilizingFusion
from moderntsf.models.dtaf.model import Model as DTAF
from moderntsf.models.dtaf.spec import ModelParameterConfig as DTAFParameters
from moderntsf.models.fredformer.model import FrequencyEqualization, split_frequency_bands
from moderntsf.models.fredformer.model import Model as Fredformer
from moderntsf.models.fredformer.spec import ModelParameterConfig as FredformerParameters
from moderntsf.models.hdmixer.model import HierarchicalDependencyBlock, LengthExtendablePatcher
from moderntsf.models.hdmixer.model import Model as HDMixer
from moderntsf.models.hdmixer.spec import ModelParameterConfig as HDMixerParameters
from moderntsf.models.micn.model import IsometricConvolutionBranch, MultiScaleDecomposition
from moderntsf.models.micn.model import Model as MICN
from moderntsf.models.micn.spec import ModelParameterConfig as MICNParameters
from moderntsf.models.msgnet.model import AdaptiveMixHopGraph, MultiScaleGraphBlock
from moderntsf.models.msgnet.model import Model as MSGNet
from moderntsf.models.msgnet.spec import ModelParameterConfig as MSGNetParameters
from moderntsf.models.moderntcn.model import LargeKernelDepthwiseConv, ModernTCNBlock
from moderntsf.models.moderntcn.model import Model as ModernTCN
from moderntsf.models.moderntcn.spec import ModelParameterConfig as ModernTCNParameters


class PaperStructureTests(unittest.TestCase):
    def test_dtaf_causal_tfs_and_frequency_difference(self):
        torch.manual_seed(1)
        tfs = TemporalStabilizingFusion(4, 2, 3, 0).eval()
        history = torch.randn(1, 6, 4)
        changed = history.clone(); changed[:, 4:] += 100
        torch.testing.assert_close(tfs(history)[:, :4], tfs(changed)[:, :4])
        fwm = FrequencyWaveModeling(4, 1, 1, 0)
        spectrum, mask = fwm.spectral_mask(torch.sin(torch.arange(8.0))[None, :, None].repeat(1, 1, 4))
        self.assertEqual(mask.shape, spectrum.shape[:2])
        self.assertTrue(mask[:, 0].all())
        self.assertEqual(int(mask.sum()), 2)

    def test_fredformer_equalizes_each_frequency_band(self):
        spectrum = torch.complex(torch.randn(2, 3, 9), torch.randn(2, 3, 9))
        bands, bins = split_frequency_bands(spectrum, 4)
        normalized, energy = FrequencyEqualization()(bands)
        self.assertEqual(bins, 9)
        self.assertEqual(tuple(bands.shape), (2, 3, 3, 4))
        torch.testing.assert_close(normalized.abs().square().mean((-1, -2)), torch.ones(2, 3), atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(normalized * energy, bands)

    def test_hdmixer_extendable_patches_and_four_axis_hierarchy(self):
        patcher = LengthExtendablePatcher(16, 4, 4, 0.25)
        grid = patcher.sampling_grid(torch.zeros(2, 16))
        self.assertEqual(tuple(grid.shape), (2, 4, 4))
        self.assertTrue((grid[:, :, 1:] >= grid[:, :, :-1]).all())
        block = HierarchicalDependencyBlock(2, 4, 4, 8, 16, 0)
        self.assertEqual(tuple(block(torch.randn(2, 2, 4, 4, 8)).shape), (2, 2, 4, 4, 8))

    def test_micn_multiscale_decomposition_and_isometric_round_trip(self):
        seasonal, trend = MultiScaleDecomposition((5, 9))(torch.randn(2, 16, 3))
        self.assertEqual(seasonal.shape, trend.shape)
        branch = IsometricConvolutionBranch(8, 4, 0)
        self.assertEqual(tuple(branch(torch.randn(2, 17, 8)).shape), (2, 17, 8))
        self.assertEqual(branch.local.stride, (4,))
        self.assertEqual(branch.restore.stride, (4,))

    def test_msgnet_scale_specific_adjacency_and_mixhop(self):
        graph = AdaptiveMixHopGraph(3, 4, 2, 0.3, 0)
        torch.testing.assert_close(graph.adjacency().sum(-1), torch.ones(3))
        self.assertEqual(graph.projection.in_features, 3)
        block = MultiScaleGraphBlock(3, 2, 8, 2, 4, 2, 0.3, 0)
        self.assertIsNot(block.branches[0].graph.source, block.branches[1].graph.source)
        self.assertEqual(tuple(block(torch.randn(2, 16, 3)).shape), (2, 16, 3))

    def test_moderntcn_large_kernel_and_two_grouped_ffns(self):
        block = ModernTCNBlock(3, 4, 2, 7, 3, 0)
        self.assertIsInstance(block.temporal, LargeKernelDepthwiseConv)
        self.assertEqual(block.temporal.large.groups, 12)
        self.assertEqual(block.variable_ffn[0].groups, 3)
        self.assertEqual(block.feature_ffn[0].groups, 4)
        self.assertEqual(tuple(block(torch.randn(2, 3, 4, 8)).shape), (2, 3, 4, 8))


class SchemaTests(unittest.TestCase):
    def test_invalid_architecture_constraints(self):
        invalid = (
            lambda: DTAFParameters(enc_in=2, d_model=7, heads=2),
            lambda: FredformerParameters(enc_in=2, model_width=7, heads=2),
            lambda: HDMixerParameters(enc_in=2, deform_range=2),
            lambda: MICNParameters(enc_in=2, conv_kernel=[4, 4]),
            lambda: MSGNetParameters(enc_in=2, d_model=7, n_heads=2),
            lambda: ModernTCNParameters(enc_in=2, large_size=[4]),
        )
        for factory in invalid:
            with self.subTest(factory=factory), self.assertRaises(ValidationError): factory()


class RuntimeTests(unittest.TestCase):
    @staticmethod
    def factories(length=16, pred=4):
        return {
            "DTAF": lambda: DTAF(length,pred,2,d_model=8,e_layers=1,patch_len=4,stride=2,heads=2,dropout=0,expert_num=2,expert_hidden=4,top_k=1),
            "Fredformer": lambda: Fredformer(length,pred,2,band_width=4,model_width=8,depth=1,heads=2,feedforward=16,dropout=0),
            "HDMixer": lambda: HDMixer(length,pred,2,d_model=8,d_ff=16,e_layers=1,patch_len=4,stride=4,dropout=0),
            "MICN": lambda: MICN(length,pred,2,d_model=8,d_layers=1,conv_kernel=(4,8),dropout=0),
            "MSGNet": lambda: MSGNet(length,pred,enc_in=2,d_model=8,e_layers=1,n_heads=2,top_k=2,dropout=0,gcn_depth=2,node_dim=4),
            "ModernTCN": lambda: ModernTCN(length,pred,2,dims=(8,),num_blocks=(1,),large_size=(5,),small_size=(3,),patch_size=4,patch_stride=2,dropout=0),
        }

    @staticmethod
    def call(model, values, changed_marks=False):
        marks = torch.randn(values.shape[0], values.shape[1], 6)
        if changed_marks: marks += 100
        return model(values, marks, None, None)

    def test_forward_backward_gradients_state_and_boundaries(self):
        torch.manual_seed(25)
        for name, factory in self.factories().items():
            with self.subTest(model=name):
                model = factory().cpu()
                values = torch.randn(2,16,2,requires_grad=True)
                output = self.call(model, values)
                self.assertEqual(tuple(output.shape),(2,4,2)); self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward(); self.assertIsNotNone(values.grad)
                for parameter_name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, f"{name}:{parameter_name}")
                    self.assertGreater(parameter.grad.abs().max(), 0, f"{name}:{parameter_name}")
                model.eval(); expected=self.call(model,values.detach())
                clone=factory().eval(); clone.load_state_dict(copy.deepcopy(model.state_dict()))
                torch.testing.assert_close(self.call(clone,values.detach()),expected)
                torch.testing.assert_close(self.call(model,values.detach(),True),expected)
                self.assertEqual(tuple(self.call(model,torch.randn(1,16,2)).shape),(1,4,2))
                with self.assertRaises(ValueError): self.call(model,torch.randn(1,15,2))


if __name__ == "__main__": unittest.main()

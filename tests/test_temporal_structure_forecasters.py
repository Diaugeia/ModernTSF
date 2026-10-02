"""Paper-structure and runtime tests for the final temporal implementation batch."""
from __future__ import annotations

import copy
import unittest
import torch
from pydantic import ValidationError

from tsflab.models.koopa.model import FourierDynamicsSplit, estimate_operator, Model as Koopa
from tsflab.models.koopa.spec import ModelParameterConfig as KoopaParameters
from tsflab.models.latenttsf.model import LatentStateAutoencoder, latent_alignment_loss, Model as LatentTSF
from tsflab.models.latenttsf.spec import ModelParameterConfig as LatentParameters
from tsflab.models.softs.model import SeriesCoreFusion, Model as SOFTS
from tsflab.models.softs.spec import ModelParameterConfig as SOFTSParameters
from tsflab.models.srsnet.model import SelectivePatching, DynamicReassembly, Model as SRSNet
from tsflab.models.srsnet.spec import ModelParameterConfig as SRSParameters
from tsflab.models.sumba.model import StructuredMatrixBasis, Model as Sumba
from tsflab.models.sumba.spec import ModelParameterConfig as SumbaParameters
from tsflab.models.timealign.model import DistributionAlignment, Model as TimeAlign
from tsflab.models.timealign.spec import ModelParameterConfig as TimeAlignParameters


class PaperStructureTests(unittest.TestCase):
    def test_koopa_fourier_partition_and_dmd_transition(self):
        values = torch.randn(2, 16, 3)
        variant, invariant = FourierDynamicsSplit(0.25)(values)
        torch.testing.assert_close(variant + invariant, values)
        states = torch.tensor([[[1.0], [2.0], [4.0], [8.0]]])
        torch.testing.assert_close(estimate_operator(states), torch.tensor([[[2.0]]]), atol=2e-4, rtol=2e-4)

    def test_softs_star_has_one_normalized_global_core(self):
        star = SeriesCoreFusion(8, 4)
        tokens = torch.randn(2, 5, 8)
        core, weights = star.aggregate(tokens)
        self.assertEqual(tuple(core.shape), (2, 1, 4))
        torch.testing.assert_close(weights.sum(1), torch.ones(2, 1))
        self.assertEqual(tuple(star(tokens).shape), tuple(tokens.shape))

    def test_srs_selects_and_softly_reassembles_patches(self):
        patches = torch.randn(2, 3, 4, 5)
        selected, scores = SelectivePatching(5, 8, 4, 2.0, 0)(patches)
        assignment = DynamicReassembly(2.0).assignment(scores)
        self.assertEqual(tuple(selected.shape), (2, 3, 4, 8))
        torch.testing.assert_close(assignment.sum(-2), torch.ones(2, 3, 4))
        self.assertEqual(tuple(DynamicReassembly(2.0)(selected, scores).shape), tuple(selected.shape))

    def test_sumba_bases_and_dynamic_graph_are_convex(self):
        basis = StructuredMatrixBasis(3, 8, 4, 2)
        matrices = basis.matrices()
        torch.testing.assert_close(matrices.sum(-1), torch.ones(4, 3))
        graph, weights = basis(torch.randn(2, 6, 3, 8))
        torch.testing.assert_close(weights.sum(-1), torch.ones(2))
        torch.testing.assert_close(graph.sum(-1), torch.ones(2, 3))
        self.assertGreaterEqual(float(basis.diversity_penalty()), 0.0)

    def test_latenttsf_expands_states_and_uses_latent_alignment(self):
        autoencoder = LatentStateAutoencoder(3, 8, 12)
        values = torch.randn(2, 6, 3)
        states = autoencoder.encode(values)
        self.assertEqual(tuple(states.shape), (2, 6, 8))
        self.assertEqual(tuple(autoencoder.decode(states).shape), tuple(values.shape))
        torch.testing.assert_close(latent_alignment_loss(states, states), torch.zeros(()), atol=1e-6, rtol=0)

    def test_timealign_has_local_and_global_terms(self):
        alignment = DistributionAlignment(0.0, 0.0, True, True)
        prediction, target = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 4, 8)
        terms = alignment.terms(prediction, target)
        self.assertEqual(set(terms), {"local", "global"})
        self.assertTrue(all(value.ndim == 0 and value >= 0 for value in terms.values()))


class SchemaTests(unittest.TestCase):
    def test_invalid_parameters_are_rejected(self):
        invalid = (
            lambda: KoopaParameters(enc_in=2, alpha=0),
            lambda: LatentParameters(enc_in=2, ae_loss="bad"),
            lambda: SOFTSParameters(enc_in=2, activation="silu"),
            lambda: SRSParameters(enc_in=2, stride=0),
            lambda: SumbaParameters(enc_in=2, basis_count=0),
            lambda: TimeAlignParameters(enc_in=2, patch_num=0),
        )
        for factory in invalid:
            with self.subTest(factory=factory), self.assertRaises(ValidationError):
                factory()


class RuntimeTests(unittest.TestCase):
    @staticmethod
    def factories(length=16, pred=4):
        return {
            "Koopa": lambda: Koopa(length, pred, 2, seg_len=4, dynamic_dim=8, hidden_dim=8, hidden_layers=1, num_blocks=1, alpha=0.25),
            "LatentTSF": lambda: LatentTSF(length, pred, 2, d_model=8, d_ff=16, kernel_size=3, ae_train_epochs=0),
            "SOFTS": lambda: SOFTS(length, pred, 2, d_model=8, d_core=4, d_ff=16, e_layers=1, dropout=0),
            "SRSNet": lambda: SRSNet(length, pred, 2, d_model=8, patch_len=4, stride=4, hidden_size=4, dropout=0, head_dropout=0),
            "Sumba": lambda: Sumba(length, pred, 2, d_model=8, basis_count=3, basis_rank=4, temporal_kernels=(2,3), depth=1, diffusion_steps=2, dropout=0),
            "TimeAlign": lambda: TimeAlign(length, pred, 2, patch_num=2, d_model=8, d_ff=16, e_layers=1, dropout=0),
        }

    @staticmethod
    def call(model, values, target=False, changed_marks=False):
        marks = torch.randn(values.size(0), values.size(1), 6)
        if changed_marks:
            marks += 100
        if target and callable(getattr(model, "training_objective", None)):
            future = torch.linspace(
                -1, 1, values.size(0) * 4 * values.size(2), device=values.device
            ).reshape(values.size(0), 4, values.size(2))
            output, objective, _ = model.training_objective(values, future)
            return output, objective
        output = model(values, marks, None, None)
        objective = output.square().mean()
        return output, objective

    def test_forward_backward_gradients_state_and_boundaries(self):
        torch.manual_seed(33)
        for name, factory in self.factories().items():
            with self.subTest(model=name):
                model = factory().cpu()
                values = torch.randn(2, 16, 2, requires_grad=True)
                output, objective = self.call(model, values, target=True)
                self.assertEqual(tuple(output.shape), (2, 4, 2))
                self.assertTrue(torch.isfinite(output).all())
                objective.backward()
                self.assertIsNotNone(values.grad)
                for parameter_name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    self.assertIsNotNone(parameter.grad, f"{name}:{parameter_name}")
                    self.assertTrue(torch.isfinite(parameter.grad).all(), f"{name}:{parameter_name}")
                    self.assertGreater(parameter.grad.abs().max(), 0, f"{name}:{parameter_name}")
                model.eval()
                expected, _ = self.call(model, values.detach())
                clone = factory().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()))
                actual, _ = self.call(clone, values.detach())
                torch.testing.assert_close(actual, expected)
                changed, _ = self.call(model, values.detach(), changed_marks=True)
                torch.testing.assert_close(changed, expected)
                batch_one, _ = self.call(model, torch.randn(1, 16, 2))
                self.assertEqual(tuple(batch_one.shape), (1, 4, 2))
                with self.assertRaises(ValueError):
                    self.call(model, torch.randn(1, 15, 2))


if __name__ == "__main__":
    unittest.main()

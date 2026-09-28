"""Equation, structure, schema, and runtime tests for six SSM/sequence implementations."""

from __future__ import annotations

import copy
import unittest

import torch
from pydantic import ValidationError

from moderntsf.models._components.mamba import MambaBlock
from moderntsf.models.bimamba.model import MambaPlus, SeriesRelationDecider, patchify
from moderntsf.models.bimamba.model import Model as BiMamba
from moderntsf.models.bimamba.spec import ModelParameterConfig as BiMambaParameters
from moderntsf.models.mambasimple.model import Model as MambaSimple
from moderntsf.models.mambasimple.spec import ModelParameterConfig as MambaSimpleParameters
from moderntsf.models.reformer.model import LSHSelfAttention, ReversibleBlock
from moderntsf.models.reformer.model import Model as Reformer
from moderntsf.models.reformer.spec import ModelParameterConfig as ReformerParameters
from moderntsf.models.s4.model import DiagonalSSMKernel, zoh_discretize_diagonal
from moderntsf.models.s4.model import Model as S4
from moderntsf.models.s4.spec import ModelParameterConfig as S4Parameters
from moderntsf.models.s_mamba.model import InvertedTokenization
from moderntsf.models.s_mamba.model import Model as SMamba
from moderntsf.models.s_mamba.spec import ModelParameterConfig as SMambaParameters
from moderntsf.models.scinet.model import Model as SCINet
from moderntsf.models.scinet.model import SCIInteraction, SCITree, interleave
from moderntsf.models.scinet.spec import ModelParameterConfig as SCINetParameters


def marks(batch, length):
    result = torch.zeros(batch, length, 6)
    result[..., 0] = 2024
    result[..., 1] = 1
    result[..., 2] = torch.arange(length) + 1
    result[..., 3] = torch.arange(length) % 7
    result[..., 4] = torch.arange(length) % 24
    return result


class Constant(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, values):
        return torch.full_like(values, self.value)


class PaperStructureTests(unittest.TestCase):
    def test_bimamba_patch_sra_and_complementary_gate(self):
        values = torch.arange(1.0, 9.0).reshape(1, 8, 1)
        patches = patchify(values, 4, 2)
        self.assertEqual(tuple(patches.shape), (1, 1, 3, 4))
        torch.testing.assert_close(
            patches[0, 0, -1], torch.tensor([5.0, 6.0, 7.0, 8.0])
        )
        correlated = torch.arange(8.0).view(1, 8, 1).repeat(1, 1, 2)
        independent = torch.stack(
            (torch.arange(8.0), torch.tensor([0.0, 7.0, 1.0, 6.0, 2.0, 5.0, 3.0, 4.0])),
            -1,
        ).unsqueeze(0)
        decider = SeriesRelationDecider(0.5)
        self.assertGreater(decider(correlated).item(), decider(independent).item())
        block = MambaPlus(4, 2, 1, 1)
        block.scan = Constant(3.0)
        block.norm = torch.nn.Identity()
        with torch.no_grad():
            block.new_gate.weight.zero_()
            block.new_gate.bias.zero_()
        fixture = torch.ones(1, 2, 4)
        torch.testing.assert_close(block(fixture), torch.full_like(fixture, 2.0))

    def test_s_mamba_token_axis_and_shared_canonical_block(self):
        tokens = InvertedTokenization(8, 4, 0.0)(torch.randn(2, 8, 3))
        self.assertEqual(tuple(tokens.shape), (2, 3, 4))
        model = SMamba(
            8,
            3,
            2,
            d_model=8,
            d_state=4,
            d_ff=16,
            e_layers=1,
            d_conv=2,
            expand=1,
            dropout=0.0,
        )
        self.assertIsInstance(model.layers[0].forward_scan, MambaBlock)
        self.assertIsInstance(model.layers[0].backward_scan, MambaBlock)

    def test_s4_zero_order_hold_and_impulse_recurrence(self):
        a = torch.tensor([[-2.0 + 0.0j]])
        b = torch.tensor([[3.0 + 0.0j]])
        dt = torch.tensor([[0.25]])
        a_bar, b_bar = zoh_discretize_diagonal(a, b, dt)
        torch.testing.assert_close(a_bar, torch.exp(dt * a))
        torch.testing.assert_close(b_bar, (torch.exp(dt * a) - 1) / a * b)
        kernel = DiagonalSSMKernel(1, 2)
        generated = kernel(5)
        ca, cb, cc = kernel.continuous_parameters()
        da, db = zoh_discretize_diagonal(ca, cb, kernel.log_dt.exp().unsqueeze(-1))
        state = torch.zeros_like(db)
        explicit = []
        for step in range(5):
            state = da * state + (db if step == 0 else 0)
            explicit.append((2 * (cc * state).sum(-1).real).item())
        torch.testing.assert_close(generated.flatten(), torch.tensor(explicit))

    def test_reformer_sparse_causal_and_reversible_contracts(self):
        attention = LSHSelfAttention(
            8, heads=2, bucket_size=2, n_hashes=2, causal=True, max_sequence_length=8
        ).eval()
        values = torch.randn(1, 8, 8)
        fixed_qk = torch.randn(1, 8, 8)
        changed = values.clone()
        changed[:, 5:] += 100
        original_output = attention(values, fixed_qk)
        changed_output = attention(changed, fixed_qk)
        torch.testing.assert_close(original_output[:, :5], changed_output[:, :5])
        # Hashes may change for future tokens too; bucket-local chunk boundaries
        # still preserve the observable causal contract for past positions.
        original_output = attention(values)
        changed_output = attention(changed)
        torch.testing.assert_close(original_output[:, :5], changed_output[:, :5])
        self.assertLessEqual(attention.last_candidate_width, 4)
        block = ReversibleBlock(8, 2, 16, 2, 2, 0.0, False, 8).eval()
        hidden = torch.randn(2, 8, 16)
        torch.testing.assert_close(
            block.inverse(block(hidden)), hidden, rtol=1e-5, atol=1e-5
        )

    def test_scinet_interaction_equations_and_recursive_tree(self):
        interaction = SCIInteraction(1, 2, 3, 0.0)
        interaction.phi, interaction.psi = Constant(0.0), Constant(0.0)
        interaction.rho, interaction.eta = Constant(0.5), Constant(0.25)
        values = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        even, odd = interaction(values)
        torch.testing.assert_close(even, torch.tensor([[[0.75], [2.75]]]))
        torch.testing.assert_close(odd, torch.tensor([[[2.5], [4.5]]]))
        torch.testing.assert_close(interleave(values[:, ::2], values[:, 1::2]), values)
        tree = SCITree(1, 3, 2, 3, 0.0)
        self.assertEqual(
            sum(isinstance(module, SCIInteraction) for module in tree.modules()), 7
        )


class SchemaTests(unittest.TestCase):
    def test_invalid_architecture_constraints(self):
        invalid = (
            lambda: BiMambaParameters(enc_in=2, c_out=1),
            lambda: MambaSimpleParameters(enc_in=2, c_out=1),
            lambda: SMambaParameters(enc_in=2, activation="silu"),
            lambda: S4Parameters(enc_in=2, d_state=3),
            lambda: ReformerParameters(enc_in=2, d_model=10, n_heads=2),
            lambda: SCINetParameters(enc_in=2, num_stacks=4),
        )
        for factory in invalid:
            with self.subTest(factory=factory), self.assertRaises(ValidationError):
                factory()


class RuntimeTests(unittest.TestCase):
    @staticmethod
    def factories(length=8, pred=3):
        return {
            "BiMamba": lambda: BiMamba(
                length,
                pred,
                2,
                d_model=8,
                d_state=4,
                e_layers=1,
                expand=1,
                d_conv=2,
                dropout=0.0,
                patch_len=min(4, length),
                stride=2,
                d_ff=16,
            ),
            "MambaSimple": lambda: MambaSimple(
                length,
                pred,
                2,
                d_model=8,
                d_state=4,
                e_layers=1,
                expand=1,
                d_conv=2,
                dropout=0.0,
            ),
            "S_Mamba": lambda: SMamba(
                length,
                pred,
                2,
                d_model=8,
                d_state=4,
                d_ff=16,
                e_layers=1,
                d_conv=2,
                expand=1,
                dropout=0.0,
            ),
            "S4": lambda: S4(
                length, pred, 2, d_model=8, d_state=4, e_layers=1, dropout=0.0
            ),
            "Reformer": lambda: Reformer(
                length,
                pred,
                2,
                d_model=16,
                n_heads=2,
                e_layers=1,
                d_ff=16,
                dropout=0.0,
                bucket_size=2,
                n_hashes=2,
            ),
            "SCINet": lambda: SCINet(
                length,
                pred,
                2,
                num_stacks=2,
                num_levels=2,
                hidden_size=4,
                kernel_size=3,
                dropout=0.0,
            ),
        }

    @staticmethod
    def call(model, name, values, changed_marks=False):
        encoder_marks = marks(values.shape[0], values.shape[1])
        decoder_marks = marks(values.shape[0], model.pred_len)
        if changed_marks:
            encoder_marks[..., 4] += 9
            decoder_marks[..., 4] += 9
        return model(
            values,
            encoder_marks,
            values.new_zeros(values.shape[0], model.pred_len, values.shape[2]),
            decoder_marks,
        )

    def test_forward_backward_active_gradients_round_trip_and_contracts(self):
        torch.manual_seed(7727)
        for name, factory in self.factories().items():
            with self.subTest(model=name):
                model = factory().cpu()
                values = torch.randn(2, 8, 2, requires_grad=True)
                output = self.call(model, name, values)
                self.assertEqual(tuple(output.shape), (2, 3, 2))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(values.grad)
                for parameter_name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, f"{name}:{parameter_name}")
                    self.assertTrue(
                        torch.isfinite(parameter.grad).all(), f"{name}:{parameter_name}"
                    )
                    self.assertGreater(
                        parameter.grad.abs().max().item(),
                        0.0,
                        f"{name}:{parameter_name}",
                    )
                model.eval()
                expected = self.call(model, name, values.detach())
                clone = factory().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()))
                torch.testing.assert_close(
                    self.call(clone, name, values.detach()), expected
                )
                self.assertEqual(
                    tuple(self.call(model, name, torch.randn(1, 8, 2)).shape), (1, 3, 2)
                )
                with self.assertRaises(ValueError):
                    self.call(model, name, torch.randn(1, 7, 2))
                changed = self.call(model, name, values.detach(), True)
                if name == "Reformer":
                    self.assertFalse(torch.equal(changed, expected))
                else:
                    torch.testing.assert_close(changed, expected)

    def test_minimum_sequence_boundaries(self):
        cases = {
            "BiMamba": BiMamba(
                2,
                1,
                1,
                d_model=4,
                d_state=2,
                e_layers=1,
                expand=1,
                d_conv=1,
                dropout=0,
                patch_len=2,
                stride=1,
                d_ff=8,
            ),
            "MambaSimple": MambaSimple(
                1, 1, 1, d_model=4, d_state=2, e_layers=1, expand=1, d_conv=1, dropout=0
            ),
            "S_Mamba": SMamba(
                1, 1, 1, d_model=4, d_state=2, d_ff=8, e_layers=1, d_conv=1, dropout=0
            ),
            "S4": S4(1, 1, 1, d_model=4, d_state=2, e_layers=1, dropout=0),
            "Reformer": Reformer(
                1,
                1,
                1,
                d_model=4,
                n_heads=1,
                e_layers=1,
                d_ff=8,
                dropout=0,
                bucket_size=1,
                n_hashes=1,
            ),
            "SCINet": SCINet(
                2,
                1,
                1,
                num_stacks=1,
                num_levels=1,
                hidden_size=2,
                kernel_size=1,
                dropout=0,
            ),
        }
        for name, model in cases.items():
            length = model.seq_len
            output = self.call(model, name, torch.randn(1, length, 1))
            self.assertEqual(tuple(output.shape), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()

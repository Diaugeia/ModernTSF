"""Paper-structure and runtime tests for AdaMSHyper."""
from __future__ import annotations

import copy
import unittest

import torch

from tsflab.models.adamshyper.model import Model as AdaMSHyper


def factory(beta=0.0, dropout=0.0, **kwargs):
    return AdaMSHyper(
        24, 5, 3, window_size=(2, 2), hyper_num=(6, 4, 3), d_embed=8, eta=2, beta=beta,
        gamma=1.5, lambda_balance=0.25, const_weight=2.0, dropout=dropout, **kwargs,
    )


class PaperStructureTests(unittest.TestCase):
    def test_multi_scale_pyramid_keeps_input_and_floors_lengths(self):
        model = factory()
        self.assertEqual(model.num_nodes, (24, 12, 6))
        series = model.multi_scale(torch.randn(2, 24, 3))
        self.assertEqual([tuple(s.shape) for s in series], [(2, 24, 3), (2, 12, 3), (2, 6, 3)])

    def test_incidence_is_binary_topk_and_thresholded(self):
        torch.manual_seed(0)
        scale = factory(beta=0.0).scales[0]
        H = scale.incidence().detach()
        self.assertTrue(((H == 0) | (H == 1)).all())
        self.assertTrue((H.sum(1) <= 2).all())  # eta
        soft = torch.softmax(torch.relu(scale.node_embedding @ scale.hyper_embedding.t()), dim=-1)
        top = soft.topk(2, dim=1).indices
        expected = torch.zeros_like(soft).scatter_(1, top, 1.0) * (soft > 0).float()
        torch.testing.assert_close(H, expected)
        strict = factory(beta=0.9).scales[0]
        strict.node_embedding.data = scale.node_embedding.data.clone()
        strict.hyper_embedding.data = scale.hyper_embedding.data.clone()
        self.assertTrue((strict.incidence().detach().sum(1) <= 1).all())  # only one entry can exceed 0.9

    def test_hyperedges_are_member_means_and_losses_match_equations(self):
        torch.manual_seed(1)
        scale = factory().scales[1].eval()
        x = torch.randn(2, 12, 3)
        out, edges, valid, node_loss, hyper_loss = scale(x)
        H = scale.incidence().detach()
        v = scale.node_init(x)
        manual_node, manual_hyper = 0.0, 0.0
        for b in range(2):
            for j in range(4):
                members = H[:, j].nonzero().flatten()
                if len(members) == 0:
                    self.assertEqual(valid[j].item(), 0.0)
                    torch.testing.assert_close(edges[b, j], torch.zeros(3))
                else:
                    torch.testing.assert_close(edges[b, j], v[b, members].mean(0))
            per_node = []
            for n in range(12):
                hs = H[n].nonzero().flatten()
                per_node.append(
                    (v[b, n] - edges[b, hs]).abs().mean(-1).mean() if len(hs) else torch.zeros(())
                )
            manual_node += torch.stack(per_node).mean().item() / 2
            ids = valid.nonzero().flatten()
            total = 0.0
            for i in ids:
                for j in ids:
                    a = torch.nn.functional.cosine_similarity(edges[b, i], edges[b, j], dim=0)
                    d = (edges[b, i] - edges[b, j]).norm()
                    total += (a * d + (1 - a) * torch.clamp(scale.gamma - d, min=0)).item()
            manual_hyper += total / (len(ids) ** 2) / 2
        self.assertAlmostEqual(node_loss.item(), manual_node, places=4)
        self.assertAlmostEqual(hyper_loss.item(), manual_hyper, places=4)

    def test_intra_scale_update_is_zero_for_isolated_nodes(self):
        torch.manual_seed(3)
        scale = factory(beta=0.99).scales[0].eval()  # a softmax row rarely exceeds 0.99
        x = torch.randn(1, 24, 3)
        with torch.no_grad():
            scale.node_embedding[0] = 0.0  # uniform softmax row (1/6) is below beta
            H = scale.incidence()
            out = scale(x)[0]
        self.assertEqual(H[0].sum().item(), 0.0)
        self.assertEqual(out.shape, (1, 24, 3))
        isolated = H.sum(1) == 0
        self.assertTrue((out[0][isolated] == 0).all())

    def test_constraint_loss_becomes_aux_loss_only_while_training(self):
        torch.manual_seed(2)
        model = factory()
        x = torch.randn(2, 24, 3)
        model.train()
        model(x)
        self.assertEqual(model.aux_loss.shape, ())
        self.assertTrue(torch.isfinite(model.aux_loss))
        model.eval()
        model(x)
        self.assertIsNone(model.aux_loss)

    def test_hyperedge_attention_ignores_empty_hyperedges(self):
        model = factory().eval()
        edges = torch.randn(2, 13, 3)
        valid = torch.ones(13)
        valid[[2, 7]] = 0
        out = model.hyperedge_attention(edges, valid)
        self.assertTrue((out[:, [2, 7]] == 0).all())
        edges2 = edges.clone()
        edges2[:, [2, 7]] = 99.0
        torch.testing.assert_close(model.hyperedge_attention(edges2, valid)[:, valid > 0], out[:, valid > 0])

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            AdaMSHyper(24, 5, 3, window_size=(2, 2), hyper_num=(6, 4))
        with self.assertRaises(ValueError):
            AdaMSHyper(3, 5, 3, window_size=(2, 2), hyper_num=(6, 4, 3))


class RuntimeContractTests(unittest.TestCase):
    def test_forward_backward_round_trip_and_bounds(self):
        torch.manual_seed(7)
        x = torch.randn(4, 24, 3)
        model = factory().cpu().train()
        value = x.clone().requires_grad_(True)
        output = model(value)
        self.assertEqual(output.shape, (4, 5, 3))
        self.assertTrue(torch.isfinite(output).all())
        (output.square().mean() + model.aux_loss).backward()
        self.assertGreater(value.grad.abs().max().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        model.eval()
        clone = factory().cpu().eval()
        clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        torch.testing.assert_close(clone(x), model(x))
        self.assertEqual(model(x[:1]).shape, (1, 5, 3))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 23, 3))


if __name__ == "__main__":
    unittest.main()

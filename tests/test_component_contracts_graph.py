"""Contract and numerical-regression tests for graph-group shared components.

Covers adaptive_node_embedding_adjacency, adj_norm, diffusion_conv,
graph_spectral, graph_utils, node_visibility, regularized_adaptive_graph_conv,
deviation_memory, sparse_connection_router, and gated_dilated_conv. Reference
values live in ``tests/fixtures/components/<name>.pt``; regenerate only for a
reviewed numerical change (see ``tests/component_reference.py``).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch.nn as nn

from tests.component_reference import assert_reference
from tsflab.models._components.adaptive_node_embedding_adjacency import (
    adaptive_node_embedding_adjacency,
)
from tsflab.models._components.adj_norm import (
    gcn_norm,
    reverse_transition_matrix,
    scaled_laplacian as adj_scaled_laplacian,
    symmetric_normalized_laplacian,
    transition_matrix,
)
from tsflab.models._components.deviation_memory import (
    PrototypeMemory,
    PrototypeRetrieval,
    deviation_score,
)
from tsflab.models._components.diffusion_conv import (
    DiffusionConv2d,
    NeighborhoodConv2d,
    PointwiseProjection,
)
from tsflab.models._components.gated_dilated_conv import causal_pad, gated_dilated_conv
from tsflab.models._components.graph_spectral import (
    chebyshev_polynomials,
    chebyshev_supports,
    scaled_laplacian,
)
from tsflab.models._components.graph_utils import adj_to_supports, cheb_poly, normalize_adj_mx
from tsflab.models._components.node_visibility import (
    group_into_subgraphs,
    random_mask_tokens,
    shuffle_tokens,
    ungroup_subgraphs,
    unshuffle_tokens,
)
from tsflab.models._components.regularized_adaptive_graph_conv import (
    EfficientCosineGraphConv,
    StochasticSharedEmbedding,
)
from tsflab.models._components.sparse_connection_router import SharedSparseConnectionRouter


def _adj(n: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.random((n, n))
    a[a < 0.5] = 0.0
    np.fill_diagonal(a, 0.0)
    a[:, 0] = 0.0  # node 0 has zero in-degree; node n-1 row kept nonzero below
    a[n - 1] = 0.0  # zero-degree row
    return a


# ---------------------------------------------------------------- adaptive adjacency


def test_adaptive_adjacency_contract():
    torch.manual_seed(0)
    e = torch.randn(6, 4)
    a = adaptive_node_embedding_adjacency(e)
    assert a.shape == (6, 6) and a.dtype == e.dtype
    torch.testing.assert_close(a.sum(-1), torch.ones(6))
    assert (a >= 0).all()
    # batched self-similarity
    b = adaptive_node_embedding_adjacency(torch.randn(2, 6, 4))
    assert b.shape == (2, 6, 6)
    # dual form uses target as given (no implicit transpose)
    t = torch.randn(4, 6)
    d = adaptive_node_embedding_adjacency(e, t)
    torch.testing.assert_close(d, torch.softmax(torch.relu(e @ t), -1))
    torch.testing.assert_close(a, adaptive_node_embedding_adjacency(e, e.T))
    assert_reference("adaptive_node_embedding_adjacency", {"self": a, "dual": d})


def test_adaptive_adjacency_gradient():
    e = torch.randn(5, 3, requires_grad=True)
    t = torch.randn(3, 5, requires_grad=True)
    # softmax rows sum to 1, so use a weighted loss
    (adaptive_node_embedding_adjacency(e, t) * torch.arange(5.0)).sum().backward()
    assert e.grad.abs().sum() > 0 and t.grad.abs().sum() > 0


# ---------------------------------------------------------------- adj_norm


def test_adj_norm_contract_and_invariants():
    a = _adj()
    n = a.shape[0]
    outs = {
        "sym_lap": symmetric_normalized_laplacian(a),
        "scaled": adj_scaled_laplacian(a),
        "gcn": gcn_norm(a),
        "trans": transition_matrix(a),
        "rev": reverse_transition_matrix(a),
    }
    for name, m in outs.items():
        assert m.shape == (n, n) and m.dtype == np.float64, name
        assert np.isfinite(m).all(), name
    # zero-degree rows give zero transition rows; others sum to 1
    sums = outs["trans"].sum(1)
    deg = a.sum(1)
    np.testing.assert_allclose(sums[deg > 0], 1.0)
    assert (sums[deg == 0] == 0).all()
    np.testing.assert_allclose(outs["rev"], transition_matrix(a.T))
    np.testing.assert_allclose(adj_scaled_laplacian(a), outs["sym_lap"] - np.eye(n))
    np.testing.assert_allclose(adj_scaled_laplacian(a, 4.0), 0.5 * outs["sym_lap"] - np.eye(n))
    sym = np.maximum(a, a.T)
    lap = symmetric_normalized_laplacian(sym)
    np.testing.assert_allclose(lap, lap.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(gcn_norm(np.zeros((3, 3)))), 1.0)
    with pytest.raises(ValueError):
        gcn_norm(np.ones((2, 3)))
    assert_reference("adj_norm", {k: torch.as_tensor(v) for k, v in outs.items()})


# ---------------------------------------------------------------- diffusion_conv


def test_diffusion_conv_contract():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 6, requires_grad=True)
    sup = [torch.rand(5, 5) for _ in range(2)]
    nc = NeighborhoodConv2d()
    y = nc(x, sup[0])
    assert y.shape == x.shape and y.is_contiguous()
    torch.testing.assert_close(y, torch.einsum("ncvl,vw->ncwl", x, sup[0]))
    assert len(list(nc.parameters())) == 0 and len(nc.state_dict()) == 0

    proj = PointwiseProjection(3, 4)
    assert set(proj.state_dict()) == {"mlp.weight", "mlp.bias"}
    assert proj(x).shape == (2, 4, 5, 6)

    m = DiffusionConv2d(3, 4, dropout=0.5, support_len=2, order=2)
    assert m.mlp.mlp.weight.shape == (4, 5 * 3, 1, 1)
    m.eval()
    out = m(x, sup)
    assert out.shape == (2, 4, 5, 6) and out.dtype == x.dtype
    assert torch.equal(out, m(x, sup))  # dropout off in eval
    out.sum().backward()
    assert x.grad.abs().sum() > 0
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    with pytest.raises(ValueError):
        m(x, sup[:1])
    with pytest.raises(ValueError):
        DiffusionConv2d(3, 4, 0.0, support_len=0)
    with pytest.raises(ValueError):
        DiffusionConv2d(3, 4, 0.0, order=0)
    m.train()
    assert not torch.equal(m(x, sup), m(x, sup))  # dropout active in train
    m.eval()
    assert_reference("diffusion_conv", {"out": out})


# ---------------------------------------------------------------- graph_spectral


def test_graph_spectral_contract():
    a = _adj(6, 1)
    s = scaled_laplacian(a)
    assert s.shape == (6, 6) and s.dtype == np.float32 and np.isfinite(s).all()
    np.testing.assert_allclose(s, s.T, atol=1e-5)  # undirected symmetrises
    ev = np.linalg.eigvalsh(s.astype(np.float64))
    assert ev.min() >= -1 - 1e-4 and ev.max() <= 1 + 1e-4
    sd = scaled_laplacian(a, undirected=False)
    assert sd.dtype == np.float32 and sd.shape == (6, 6)
    assert scaled_laplacian(np.zeros((3, 3))).shape == (3, 3)
    assert scaled_laplacian(np.zeros((0, 0))).shape == (0, 0)
    with pytest.raises(ValueError):
        scaled_laplacian(np.ones((2, 3)))
    with pytest.raises(ValueError):
        scaled_laplacian(np.full((2, 2), np.nan))

    polys = chebyshev_polynomials(s, 4)
    assert polys.shape == (4, 6, 6)
    np.testing.assert_allclose(polys[0], np.eye(6))
    np.testing.assert_allclose(polys[1], s)
    np.testing.assert_allclose(polys[2], 2 * s @ s - np.eye(6), atol=1e-5)
    np.testing.assert_allclose(polys[3], 2 * s @ polys[2] - polys[1], atol=1e-5)
    assert chebyshev_polynomials(s, 1).shape == (1, 6, 6)
    with pytest.raises(ValueError):
        chebyshev_polynomials(s, 0)
    with pytest.raises(ValueError):
        chebyshev_polynomials(np.ones((2, 3)), 2)

    sup = chebyshev_supports(a, 3)
    assert sup.shape == (3, 6, 6) and sup.dtype == torch.float32
    assert_reference("graph_spectral", {"scaled": torch.as_tensor(s), "cheb": sup})


# ---------------------------------------------------------------- graph_utils


@pytest.mark.parametrize(
    "adj_type,count",
    [("normlap", 1), ("scalap", 1), ("symadj", 1), ("transition", 1),
     ("doubletransition", 2), ("identity", 1), ("origin", 1)],
)
def test_graph_utils_normalize_modes(adj_type, count):
    a = _adj()
    dense = normalize_adj_mx(a, adj_type)
    assert len(dense) == count
    for d in dense:
        assert d.shape == (5, 5) and d.dtype == np.float32 and np.isfinite(d).all()
    coo = normalize_adj_mx(a, adj_type, "coo")
    for d, c in zip(dense, coo):
        assert sp.isspmatrix_coo(c)
        np.testing.assert_allclose(c.toarray(), d)
    sup = adj_to_supports(a, adj_type)
    assert len(sup) == count and all(s.dtype == torch.float32 for s in sup)
    np.testing.assert_allclose(sup[0].numpy(), dense[0])


def test_graph_utils_values_and_errors():
    a = _adj()
    sym = normalize_adj_mx(a, "symadj")[0]
    np.testing.assert_allclose(sym, np.eye(5) - symmetric_normalized_laplacian(a), atol=1e-6)
    origin = normalize_adj_mx(a, "origin")[0]
    np.testing.assert_allclose(np.diag(origin), 1.0)
    fwd, rev = normalize_adj_mx(a, "doubletransition")
    np.testing.assert_allclose(rev, transition_matrix(a.T), atol=1e-6)
    np.testing.assert_allclose(cheb_poly(np.eye(3), 3), chebyshev_polynomials(np.eye(3), 3))
    with pytest.raises(ValueError):
        normalize_adj_mx(a, "bogus")
    with pytest.raises(ValueError):
        normalize_adj_mx(a, "origin", "csr")
    with pytest.raises(ValueError):
        normalize_adj_mx(np.ones((2, 3)), "origin")
    with pytest.raises(ValueError):
        normalize_adj_mx(np.full((2, 2), np.inf), "origin")
    assert_reference(
        "graph_utils",
        {"fwd": torch.as_tensor(fwd), "rev": torch.as_tensor(rev), "scalap": torch.as_tensor(normalize_adj_mx(a, "scalap")[0])},
    )


# ---------------------------------------------------------------- node_visibility


def test_node_visibility_contract():
    torch.manual_seed(0)
    x = torch.randn(2, 10, 3, requires_grad=True)
    gen = torch.Generator().manual_seed(7)
    kept, idx = random_mask_tokens(x, 0.4, gen)
    assert kept.shape == (2, 6, 3) and idx.shape == (6,) and kept.dtype == x.dtype
    assert (idx[1:] > idx[:-1]).all()
    torch.testing.assert_close(kept, x[:, idx])
    same, full = random_mask_tokens(x, 0.0)
    assert same is x and torch.equal(full, torch.arange(10))
    assert random_mask_tokens(x, 0.99)[0].shape[1] == 1
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError):
            random_mask_tokens(x, bad)
    kept2, idx2 = random_mask_tokens(x, 0.4, torch.Generator().manual_seed(7))
    assert torch.equal(idx, idx2)

    sh, perm = shuffle_tokens(x, torch.Generator().manual_seed(3))
    assert sh.shape == x.shape and perm.shape == (2, 10)
    assert torch.equal(perm.sort(1).values, torch.arange(10).expand(2, -1))
    torch.testing.assert_close(unshuffle_tokens(sh, perm), x)

    g, ng, ol = group_into_subgraphs(x, 4)
    assert g.shape == (2 * 3, 4, 3) and ng == 3 and ol == 10
    assert (g.reshape(2, 12, 3)[:, 10:] == 0).all()
    ug = ungroup_subgraphs(g, ng, ol, 4)
    torch.testing.assert_close(ug, x)
    g2, ng2, ol2 = group_into_subgraphs(x, 10)
    assert g2 is x and ng2 == 1 and ol2 == 10
    assert ungroup_subgraphs(g2, ng2, ol2, 10) is g2
    with pytest.raises(ValueError):
        group_into_subgraphs(x, 0)
    # gradient flows through the full pipeline
    ungroup_subgraphs(group_into_subgraphs(unshuffle_tokens(sh, perm), 4)[0], 3, 10, 4).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    assert_reference("node_visibility", {"kept": kept.detach(), "idx": idx, "shuffled": sh.detach(), "perm": perm, "grouped": g.detach()})


# ---------------------------------------------------------------- regularized adaptive graph conv


def test_stochastic_shared_embedding():
    torch.manual_seed(0)
    emb = torch.randn(8, 3, requires_grad=True)
    m = StochasticSharedEmbedding(0.5)
    assert len(m.state_dict()) == 0
    m.eval()
    assert m(emb) is emb
    m.train()
    assert StochasticSharedEmbedding(0.0)(emb) is emb
    out = m(emb)
    assert out.shape == emb.shape and out.dtype == emb.dtype
    for row in out:  # every row is some original row
        assert any(torch.equal(row, r) for r in emb)
    full = StochasticSharedEmbedding(1.0)
    out1 = full(emb)
    out1.sum().backward()
    assert emb.grad is not None
    with pytest.raises(ValueError):
        StochasticSharedEmbedding(1.5)
    torch.manual_seed(5)
    ref = StochasticSharedEmbedding(0.5)(emb.detach())
    assert_reference("stochastic_shared_embedding", {"out": ref})


def test_efficient_cosine_graph_conv():
    torch.manual_seed(0)
    m = EfficientCosineGraphConv(hidden_dim=4, spatial_dim=3, order=2)
    assert set(m.state_dict()) == {"gate_weight", "filter_weight", "out_weight"}
    assert m.out_weight.shape == (4, 12)
    emb = torch.randn(6, 3, requires_grad=True)
    x = torch.randn(2, 6, 4, requires_grad=True)
    sup = m.compute_support(emb)
    assert sup.shape == (6, 3)
    torch.testing.assert_close(sup.norm(dim=1), torch.ones(6), atol=1e-5, rtol=1e-5)
    y = m(x, emb)
    assert y.shape == x.shape and y.dtype == x.dtype
    # kernelised hop equals explicit D^-1 A x with A = S S^T
    s = sup.detach()
    a = s @ s.T
    hop = a @ x.detach() / (a.sum(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(m._kernelized_hop(x.detach(), s), hop, atol=1e-4, rtol=1e-4)
    y.sum().backward()
    assert x.grad.abs().sum() > 0 and emb.grad.abs().sum() > 0
    assert all(p.grad is not None for p in m.parameters())
    with pytest.raises(ValueError):
        m(x, torch.randn(6, 2))
    with pytest.raises(ValueError):
        m(torch.randn(2, 6, 5), emb)
    with pytest.raises(ValueError):
        EfficientCosineGraphConv(4, 3, order=0)
    with pytest.raises(ValueError):
        EfficientCosineGraphConv(0, 3)
    assert isinstance(EfficientCosineGraphConv(4, 3, dropout=0.1).dropout, nn.Dropout)
    assert_reference("regularized_adaptive_graph_conv", {"support": sup.detach(), "out": y.detach()})


# ---------------------------------------------------------------- deviation_memory


def test_prototype_memory_contract():
    torch.manual_seed(0)
    m = PrototypeMemory(query_dim=5, prototype_dim=4, num_prototypes=6)
    assert set(m.state_dict()) == {"prototypes", "query_proj"}
    h = torch.randn(2, 3, 5, requires_grad=True)
    r = m(h)
    assert isinstance(r, PrototypeRetrieval)
    assert r.value.shape == r.query.shape == r.nearest.shape == r.second_nearest.shape == (2, 3, 4)
    assert r.indices.shape == (2, 3, 2) and r.indices.dtype == torch.long
    assert (r.indices[..., 0] != r.indices[..., 1]).all()
    scores = torch.softmax(r.query @ m.prototypes.T, -1)
    top = scores.topk(2, -1).indices
    assert torch.equal(top, r.indices)
    torch.testing.assert_close(r.nearest, m.prototypes[r.indices[..., 0]])
    torch.testing.assert_close(r.value, scores @ m.prototypes)
    (r.value.sum() + r.nearest.sum() + r.query.sum()).backward()
    assert h.grad.abs().sum() > 0 and all(p.grad is not None for p in m.parameters())
    assert m(torch.randn(5)).value.shape == (4,)
    with pytest.raises(ValueError):
        m(torch.randn(2, 4))
    with pytest.raises(ValueError):
        PrototypeMemory(5, 4, 1)
    with pytest.raises(ValueError):
        PrototypeMemory(0, 4, 3)
    assert_reference("deviation_memory", {"value": r.value.detach(), "query": r.query.detach(), "indices": r.indices})


def test_deviation_score():
    a = torch.tensor([[1.0, -2.0, 3.0], [0.0, 0.0, 0.0]])
    b = torch.zeros(3)
    s = deviation_score(a, b)
    assert s.shape == (2,) and (s >= 0).all()
    # NOTE: code sums |a-b| over the last axis (docstring calls it a mean).
    torch.testing.assert_close(s, torch.tensor([6.0, 0.0]))
    assert deviation_score(a, a).sum() == 0
    with pytest.raises(ValueError):
        deviation_score(a, torch.zeros(2))
    x = a.clone().requires_grad_()
    deviation_score(x, b).sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------- sparse_connection_router


def test_sparse_connection_router_contract():
    torch.manual_seed(0)
    r = SharedSparseConnectionRouter(num_positions=6, dim=8, heads=2, density=0.25)
    assert "_position_index" not in r.state_dict()
    assert "memory.weight" in r.state_dict()
    r.eval()
    c, p = r()
    assert c.shape == p.shape == (2, 6, 6) and c.dtype == torch.float32
    k = round(0.25 * 36)
    assert torch.equal(c.round(), c.round().clamp(0, 1))
    assert torch.all(c.detach().sum((1, 2)) == k)
    assert ((c.detach() == 0) | (c.detach() == 1)).all()
    assert ((p >= 0) & (p <= 1)).all()
    c2, p2 = r()
    assert torch.equal(c, c2) and torch.equal(p, p2)  # deterministic in eval
    (c * torch.rand_like(c)).sum().backward()
    assert r.memory.weight.grad.abs().sum() > 0
    assert all(q.grad is not None for q in r.pair_scorer.parameters())
    r.train()
    cs, _ = r()
    assert cs.detach().sum((1, 2)).eq(k).all()
    for kw in ({"num_positions": 0}, {"heads": 0}, {"density": 0.0}, {"density": 1.5}):
        args = {"num_positions": 4, "dim": 4, **kw}
        with pytest.raises(ValueError):
            SharedSparseConnectionRouter(**args)
    full = SharedSparseConnectionRouter(3, 4, density=1.0).eval()
    assert full()[0].detach().eq(1).all()
    assert_reference("sparse_connection_router", {"conn": c.detach(), "prob": p.detach()})


# ---------------------------------------------------------------- gated_dilated_conv


def test_causal_pad_and_gated_conv():
    x3 = torch.randn(2, 3, 7)
    x4 = torch.randn(2, 3, 5, 7)
    assert causal_pad(x3, 2, 3).shape == (2, 3, 11)
    assert causal_pad(x4, 2, 3).shape == (2, 3, 5, 11)
    assert (causal_pad(x4, 2, 3)[..., :4] == 0).all()
    torch.testing.assert_close(causal_pad(x4, 2, 3)[..., 4:], x4)
    assert causal_pad(x3, 5, 1).shape == x3.shape
    with pytest.raises(ValueError):
        causal_pad(torch.randn(2, 3), 1, 2)

    torch.manual_seed(0)
    f1, g1 = nn.Conv1d(3, 4, 2, dilation=2), nn.Conv1d(3, 4, 2, dilation=2)
    y = gated_dilated_conv(x3.requires_grad_(), f1, g1)
    assert y.shape == (2, 4, 7) and y.dtype == x3.dtype  # causal: length preserved
    assert y.abs().max() < 1
    y.sum().backward()
    assert x3.grad.abs().sum() > 0 and f1.weight.grad is not None and g1.weight.grad is not None
    # causality: perturbing a future step leaves earlier outputs unchanged
    xp = x3.detach().clone()
    xp[..., 5:] += 1.0
    yp = gated_dilated_conv(xp, f1, g1)
    torch.testing.assert_close(yp[..., :5], y.detach()[..., :5])

    f2, g2 = nn.Conv2d(3, 4, (1, 3), dilation=(1, 2)), nn.Conv2d(3, 4, (1, 3), dilation=(1, 2))
    y2 = gated_dilated_conv(x4, f2, g2)
    assert y2.shape == (2, 4, 5, 7)
    assert_reference("gated_dilated_conv", {"y1": y.detach(), "y2": y2.detach()})

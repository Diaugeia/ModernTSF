"""Contract tests for attention / transformer / embedding components.

Covers differential_attention, global_patch_compression_attention,
self_attention_family, topk_expert_attention, topk_expert_router,
transformer_encdec, tst_transformer, patchtst, and embed: documented shapes,
invariants, gradient flow, and seeded numerical references stored in
``tests/fixtures/components/<component>.pt`` (regenerate only for intentional
numerical changes with ``TSFLAB_REGEN_COMPONENT_FIXTURES=1``).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.component_reference import assert_reference
from tsflab.models._components.differential_attention import DifferentialAttention
from tsflab.models._components.embed import (
    DataEmbedding,
    DataEmbedding_inverted,
    DataEmbedding_wo_pos,
    FixedEmbedding,
    PatchEmbedding,
    PositionalEmbedding,
    TemporalEmbedding,
    TimeFeatureEmbedding,
    TokenEmbedding,
)
from tsflab.models._components.global_patch_compression_attention import (
    GlobalPatchCompressionAttention,
)
from tsflab.models._components.masking import TriangularCausalMask
from tsflab.models._components.patchtst import PatchTSTBackbone
from tsflab.models._components.self_attention_family import (
    AttentionLayer,
    FlashAttention,
    FlowAttention,
    FullAttention,
    ProbAttention,
    ReformerLayer,
)
from tsflab.models._components.topk_expert_attention import (
    LocalExpertRouter,
    TopKExpertAttention,
    gather_experts,
)
from tsflab.models._components.topk_expert_router import GatingMLP, topk_dense_mix
from tsflab.models._components.transformer_encdec import (
    ConvLayer,
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
)
from tsflab.models._components.tst_transformer import TSTEncoder


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


def _grads_ok(module: nn.Module, skip: tuple[str, ...] = ()) -> None:
    for name, p in module.named_parameters():
        if not p.requires_grad or any(s in name for s in skip):
            continue
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


# --------------------------------------------------------------- differential_attention
def test_differential_attention():
    _seed()
    m = DifferentialAttention(8, 2).eval()
    assert set(m.state_dict()) == {"lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2", "rms_scale"}
    assert m.rms_scale.shape == (8,) and m.lambda_q1.shape == (2, 4)
    q = torch.randn(2, 5, 2, 8, requires_grad=True)
    k = torch.randn(2, 5, 2, 8, requires_grad=True)
    v = torch.randn(2, 5, 2, 8, requires_grad=True)
    out = m(q, k, v)
    assert out.shape == (2, 5, 2, 8) and out.dtype == q.dtype
    out.square().sum().backward()
    for t in (q, k, v):
        assert t.grad is not None and torch.isfinite(t.grad).all()
    _grads_ok(m)
    with pytest.raises(ValueError):
        DifferentialAttention(7, 2)
    assert_reference("differential_attention", {"out": out})


def test_differential_attention_is_bidirectional():
    _seed()
    m = DifferentialAttention(8, 2).eval()
    q, k, v = (torch.randn(1, 4, 2, 8) for _ in range(3))
    base = m(q, k, v)
    v2 = v.clone()
    v2[:, -1] += 1.0
    assert not torch.allclose(m(q, k, v2)[:, 0], base[:, 0])


# --------------------------------------------------- global_patch_compression_attention
def test_global_patch_compression_attention():
    _seed()
    m = GlobalPatchCompressionAttention(8, 2, 16).eval()
    keys = set(m.state_dict())
    for prefix in ("compress_attn", "broadcast_attn", "compress_mlp", "broadcast_mlp",
                   "norm_compress_attn", "norm_compress_mlp", "norm_broadcast_attn", "norm_broadcast_mlp"):
        assert any(k.startswith(prefix + ".") for k in keys), prefix
    x = torch.randn(2, 3, 4, 8, requires_grad=True)
    out = m(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    # final LayerNorm: per-token zero mean (affine identity at init)
    assert torch.allclose(out.mean(-1), torch.zeros(2, 3, 4), atol=1e-5)
    out.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    _grads_ok(m)
    with pytest.raises(ValueError):
        m(torch.randn(2, 12, 8))
    with pytest.raises(ValueError):
        GlobalPatchCompressionAttention(7, 2, 16)
    assert_reference("global_patch_compression_attention", {"out": out})


def test_global_patch_compression_last_patch_is_query():
    _seed()
    m = GlobalPatchCompressionAttention(8, 2, 16).eval()
    x = torch.randn(1, 2, 4, 8)
    y = x.clone()
    y[:, :, 0] += 1.0
    assert not torch.allclose(m(x), m(y))


# --------------------------------------------------------------- self_attention_family
def _qkv(b=2, l=6, h=2, e=4):
    return tuple(torch.randn(b, l, h, e) for _ in range(3))


def test_full_attention_matches_equation_and_mask():
    _seed()
    q, k, v = _qkv()
    core = FullAttention(mask_flag=True, attention_dropout=0.0, output_attention=True).eval()
    out, attn = core(q, k, v, None)
    assert out.shape == q.shape and attn.shape == (2, 2, 6, 6)
    scores = torch.einsum("blhe,bshe->bhls", q, k) / 2.0
    scores = scores.masked_fill(TriangularCausalMask(2, 6).mask, float("-inf"))
    expect = torch.softmax(scores, -1)
    torch.testing.assert_close(attn, expect)
    assert torch.allclose(attn.sum(-1), torch.ones(2, 2, 6), atol=1e-6)
    assert torch.all(attn.triu(1) == 0)
    full, attn2 = FullAttention(mask_flag=False, attention_dropout=0.0, output_attention=True).eval()(q, k, v, None)
    assert (attn2.triu(1) > 0).any()
    none_attn = FullAttention(mask_flag=False, output_attention=False).eval()(q, k, v, None)[1]
    assert none_attn is None
    assert_reference("self_attention_family_full", {"out": out, "attn": attn, "full": full})


def test_attention_layer_shapes_params_grads():
    _seed()
    layer = AttentionLayer(FullAttention(mask_flag=False, attention_dropout=0.0), 8, 2).eval()
    assert set(layer.state_dict()) == {
        f"{p}_projection.{w}" for p in ("query", "key", "value", "out") for w in ("weight", "bias")
    }
    x = torch.randn(2, 5, 8, requires_grad=True)
    kv = torch.randn(2, 7, 8)
    out, attn = layer(x, kv, kv, None)
    assert out.shape == (2, 5, 8) and attn is None
    out.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    _grads_ok(layer)
    assert_reference("self_attention_family_layer", {"out": out})


def test_flow_attention():
    _seed()
    q, k, v = _qkv()
    core = FlowAttention(0.0).eval()
    assert list(core.state_dict()) == []
    q.requires_grad_(True)
    out, attn = core(q, k, v, None)
    assert out.shape == q.shape and attn is None and torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(q.grad).all()
    assert_reference("self_attention_family_flow", {"out": out})


def test_flash_attention_matches_full_unmasked():
    _seed()
    q, k, v = _qkv()
    out, attn = FlashAttention(mask_flag=False, attention_dropout=0.0).eval()(q, k, v, None)
    assert attn is None and out.shape == q.shape
    ref, _ = FullAttention(mask_flag=False, attention_dropout=0.0).eval()(q, k, v, None)
    torch.testing.assert_close(out.detach(), ref, atol=1e-5, rtol=1e-4)
    pad = torch.tensor([[1, 1, 1, 1, 0, 0], [1] * 6])
    masked, _ = FlashAttention(attention_dropout=0.0).eval()(q, k, v, pad)
    v2 = v.clone()
    v2[0, 4:] += 5.0
    masked2, _ = FlashAttention(attention_dropout=0.0).eval()(q, k, v2, pad)
    torch.testing.assert_close(masked.detach()[0], masked2.detach()[0])
    assert_reference("self_attention_family_flash", {"out": out, "masked": masked})


def test_prob_attention_contract():
    _seed()
    q, k, v = _qkv(l=16)
    core = ProbAttention(mask_flag=False, factor=1, attention_dropout=0.0, output_attention=True).eval()
    assert list(core.state_dict()) == []
    out, attn = core(q, k, v, None)
    # time-first layout like the other cores / original Informer
    assert out.shape == (2, 16, 2, 4)
    assert attn.shape == (2, 2, 16, 16)
    assert torch.allclose(attn.sum(-1), torch.ones(2, 2, 16), atol=1e-5)
    cm, _ = ProbAttention(mask_flag=True, factor=1, attention_dropout=0.0).eval()(q, k, v, None)
    assert cm.shape == (2, 16, 2, 4)
    with pytest.raises(AssertionError):
        ProbAttention(mask_flag=True)(q, k[:, :8], v[:, :8], None)
    # sampling is stochastic only through torch RNG: reseeding reproduces output
    _seed(3)
    a = core(q, k, v, None)[0]
    _seed(3)
    b = core(q, k, v, None)[0]
    assert torch.equal(a, b)
    assert_reference("self_attention_family_prob", {"out": a})


def test_prob_attention_batch_or_head_one_and_layout():
    _seed()
    for b, h in ((1, 2), (2, 1), (1, 1)):
        q = torch.randn(b, 16, h, 4)
        out, _ = ProbAttention(mask_flag=False, factor=1, attention_dropout=0.0)(q, q, q, None)
        assert out.shape == (b, 16, h, 4)
    # unselected queries get mean(V) per head, laid out time-first
    q = torch.randn(2, 16, 2, 4)
    v = torch.randn(2, 16, 2, 4)
    out, _ = ProbAttention(mask_flag=False, factor=1, attention_dropout=0.0)(q, q, v, None)
    mean_v = v.mean(1, keepdim=True).expand_as(out)
    assert torch.isclose(out, mean_v, atol=1e-5).all(-1).any()


def test_reformer_layer_requires_optional_dependency():
    from tsflab.models._components import self_attention_family as fam

    if fam.LSHSelfAttention is None:
        with pytest.raises(ImportError):
            ReformerLayer(None, 8, 2)
    else:  # pragma: no cover
        assert ReformerLayer(None, 8, 2).fit_length(torch.zeros(1, 5, 8)).shape[1] % 8 == 0


# ----------------------------------------------------------------- topk_expert_attention
def test_local_expert_router_and_gather():
    _seed()
    q, k = torch.randn(3, 6, 4), torch.randn(3, 6, 4)
    router = LocalExpertRouter(4, 2)
    assert list(router.state_dict()) == []
    w, idx = router(q, k)
    assert w.shape == idx.shape == (3, 6, 2)
    assert torch.allclose(w.sum(-1), torch.ones(3, 6), atol=1e-6)
    assert idx.dtype == torch.int64 and idx.min() >= 0 and idx.max() < 6
    assert router.scale == 4**-0.5
    kv = torch.randn(3, 6, 5)
    g = gather_experts(idx, w, kv)
    assert g.shape == (3, 6, 2, 5)
    torch.testing.assert_close(g[1, 2, 0], w[1, 2, 0] * kv[1, idx[1, 2, 0]])
    with pytest.raises(RuntimeError):
        LocalExpertRouter(4, 7)(q, k)
    assert_reference("topk_expert_attention_router", {"w": w, "idx": idx.float(), "g": g})


@pytest.mark.parametrize("topk,shared", [(2, True), (2, False), (0, False), (0, True)])
def test_topk_expert_attention(topk, shared):
    _seed()
    m = TopKExpertAttention(8, num_heads=2, topk=topk, shared=shared).eval()
    assert set(m.state_dict()) == {
        "positional.weight", "positional.bias", "qkv.weight", "qkv.bias", "out_proj.weight", "out_proj.bias"
    }
    x = torch.randn(2, 6, 8, requires_grad=True)
    out = m(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    if topk:
        assert m.last_route_weight.shape == (4, 6, topk)
        assert torch.allclose(m.last_route_weight.sum(-1), torch.ones(4, 6), atol=1e-6)
    else:
        assert m.last_route_weight is None and not hasattr(m, "router")
    out.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    _grads_ok(m)
    assert_reference(f"topk_expert_attention_k{topk}_{'shared' if shared else 'plain'}", {"out": out})


def test_topk_expert_attention_errors():
    with pytest.raises(ValueError):
        TopKExpertAttention(7, num_heads=2)
    with pytest.raises(RuntimeError):
        TopKExpertAttention(8, num_heads=2, topk=9)(torch.randn(1, 6, 8))


# ------------------------------------------------------------------ topk_expert_router
def test_gating_mlp():
    _seed()
    g = GatingMLP(5, 4, 8, noisy=True)
    assert set(g.state_dict()) == {
        "network.0.weight", "network.0.bias", "network.2.weight", "network.2.bias", "noise_scale"
    }
    assert g.noise_scale.shape == (4,) and torch.all(g.noise_scale == 0)
    x = torch.randn(3, 2, 5, requires_grad=True)
    g.eval()
    w = g(x)
    assert w.shape == (3, 2, 4) and torch.allclose(w.sum(-1), torch.ones(3, 2), atol=1e-6) and (w >= 0).all()
    torch.testing.assert_close(w, torch.softmax(g.network(x), -1))
    g.train()
    _seed(1)
    n1 = g(x)
    _seed(1)
    n2 = g(x)
    assert torch.equal(n1, n2) and not torch.allclose(n1, w)
    n1.square().sum().backward()
    _grads_ok(g)
    assert list(GatingMLP(5, 4, 8, noisy=False).state_dict()) == [
        "network.0.weight", "network.0.bias", "network.2.weight", "network.2.bias"
    ]
    assert_reference("topk_expert_router_gate", {"eval": w, "noisy": n1})


def test_topk_dense_mix():
    w = torch.softmax(torch.tensor([[1.0, 2.0, 0.5, -1.0], [0.0, 0.0, 1.0, 3.0]]), -1).requires_grad_(True)
    hard = topk_dense_mix(w, 2, 0.0)
    assert torch.allclose(hard.sum(-1), torch.ones(2), atol=1e-6)
    assert (hard > 0).sum(-1).tolist() == [2, 2]
    top = w.topk(2, -1).values
    torch.testing.assert_close(hard.topk(2, -1).values, top / top.sum(-1, keepdim=True))
    soft = topk_dense_mix(w, 2, 0.1)
    assert (soft > 0).all() and torch.allclose(soft.sum(-1), torch.ones(2), atol=1e-6)
    assert soft.dtype == w.dtype
    soft[:, 0].sum().backward()
    assert torch.isfinite(w.grad).all()
    with pytest.raises(RuntimeError):
        topk_dense_mix(w, 5, 0.1)
    assert_reference("topk_expert_router_mix", {"hard": hard, "soft": soft})


# ---------------------------------------------------------------- transformer_encdec
def _attn(d=8, h=2, causal=False):
    return AttentionLayer(FullAttention(mask_flag=causal, attention_dropout=0.0), d, h)


def test_conv_layer():
    _seed()
    c = ConvLayer(8)
    assert {"down_conv.weight", "down_conv.bias", "norm.weight", "norm.running_mean"} <= set(c.state_dict())
    for length in (8, 11):
        out = c(torch.randn(2, length, 8))
        assert out.shape == (2, (length + 1) // 2 + 1, 8)
    x = torch.randn(2, 8, 8, requires_grad=True)
    c(x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_encoder_layer_and_encoder():
    _seed()
    layer = EncoderLayer(_attn(), 8, d_ff=16, dropout=0.0).eval()
    assert {k.split(".")[0] for k in layer.state_dict()} == {"attention", "conv1", "conv2", "norm1", "norm2"}
    x = torch.randn(2, 6, 8, requires_grad=True)
    out, attn = layer(x)
    assert out.shape == x.shape and attn is None
    # post-norm: output is layer-normalized
    assert torch.allclose(out.mean(-1), torch.zeros(2, 6), atol=1e-5)
    out.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    _grads_ok(layer)

    enc = Encoder([EncoderLayer(_attn(), 8, 16, 0.0) for _ in range(3)], norm_layer=nn.LayerNorm(8)).eval()
    y, attns = enc(torch.randn(2, 6, 8))
    assert y.shape == (2, 6, 8) and len(attns) == 3

    _seed()
    dist = Encoder(
        [EncoderLayer(_attn(), 8, 16, 0.0) for _ in range(3)], [ConvLayer(8) for _ in range(2)]
    ).eval()
    yd, attns = dist(torch.randn(2, 16, 8))
    assert len(attns) == 3 and yd.shape[0] == 2 and yd.shape[2] == 8 and yd.shape[1] < 16

    _seed()
    layer2 = EncoderLayer(_attn(), 8, d_ff=16, dropout=0.0, activation="gelu").eval()
    o2, _ = layer2(torch.randn(2, 6, 8))
    assert_reference("transformer_encdec_encoder", {"layer": out, "enc": y, "distil": yd, "gelu": o2})


def test_decoder_layer_and_decoder():
    _seed()
    dl = DecoderLayer(_attn(causal=True), _attn(), 8, d_ff=16, dropout=0.0).eval()
    assert {k.split(".")[0] for k in dl.state_dict()} == {
        "self_attention", "cross_attention", "conv1", "conv2", "norm1", "norm2", "norm3"
    }
    x = torch.randn(2, 5, 8, requires_grad=True)
    cross = torch.randn(2, 7, 8)
    out = dl(x, cross)
    assert out.shape == x.shape
    # causal self-attention: perturbing a later step must not change earlier
    # steps through self-attention path alone only if cross attention is token-wise;
    # cross-attention is token-wise in queries, so earlier positions are unchanged.
    x2 = x.detach().clone()
    x2[:, -1] += 1.0
    torch.testing.assert_close(dl(x2, cross)[:, 0], out[:, 0].detach(), atol=1e-5, rtol=1e-4)
    out.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    _grads_ok(dl)

    dec = Decoder([dl], norm_layer=nn.LayerNorm(8), projection=nn.Linear(8, 3)).eval()
    d = dec(x.detach(), cross)
    assert d.shape == (2, 5, 3)
    assert Decoder([dl]).eval()(x.detach(), cross).shape == (2, 5, 8)
    assert_reference("transformer_encdec_decoder", {"layer": out.detach(), "dec": d})


# ------------------------------------------------------------------ tst_transformer
@pytest.mark.parametrize("norm,pre", [("BatchNorm", False), ("LayerNorm", True)])
def test_tst_encoder(norm, pre):
    _seed()
    e = TSTEncoder(8, 2, n_layers=2, d_ff=16, norm=norm, pre_norm=pre).eval()
    keys = set(e.state_dict())
    assert any(k.startswith("layers.layers.1.") for k in keys)
    assert any(k.startswith("layers.norm.") for k in keys)
    assert ("layers.norm.norm.running_mean" in keys) == (norm == "BatchNorm")
    x = torch.randn(3, 5, 8, requires_grad=True)
    out = e(x, ignored=1)
    assert out.shape == x.shape and out.dtype == x.dtype
    e.train()
    e(x).square().sum().backward()
    assert torch.isfinite(x.grad).all()
    _grads_ok(e)
    assert_reference(f"tst_transformer_{norm.lower()}", {"out": out})


def test_tst_encoder_validation():
    with pytest.raises(ValueError):
        TSTEncoder(7, 2)
    with pytest.raises(ValueError):
        TSTEncoder(8, 2, d_k=3)
    with pytest.raises(ValueError):
        TSTEncoder(8, 2, activation="swish")
    with pytest.raises(ValueError):
        TSTEncoder(8, 2, norm="rms")
    TSTEncoder(8, 2, activation=torch.nn.functional.relu, extra_kw=1)
    assert TSTEncoder(8, 2, activation="RELU", d_k=4, d_v=4) is not None


# ------------------------------------------------------------------------- patchtst
def _backbone(**kw):
    args = dict(
        c_in=3, context_window=16, target_window=6, patch_len=4, stride=4, padding_patch="end",
        n_layers=1, d_model=8, n_heads=2, d_k=None, d_v=None, d_ff=16, activation="gelu",
        norm="BatchNorm", attn_dropout=0.0, res_dropout=0.0, ffn_dropout=0.0, proj_dropout=0.0,
        head_dropout=0.0, pre_norm=False, pe="sincos", learn_pe=True, head_type="flatten",
        individual=False, revin=True, affine=True, subtract_last=False,
    )
    args.update(kw)
    return PatchTSTBackbone(**args)


@pytest.mark.parametrize("pad,individual", [("end", False), (None, True)])
def test_patchtst_backbone(pad, individual):
    _seed()
    m = _backbone(padding_patch=pad, individual=individual).eval()
    keys = set(m.state_dict())
    for prefix in ("normalizer.", "patch_projection.", "position", "encoder.layers.", "head."):
        assert any(k.startswith(prefix) for k in keys), prefix
    x = torch.randn(2, 16, 3, requires_grad=True)
    out = m(x, "ignored marks")
    assert out.shape == (2, 6, 3) and out.dtype == x.dtype
    m.train()
    m(x).square().sum().backward()
    assert torch.isfinite(x.grad).all()
    _grads_ok(m)
    m.eval()
    assert_reference(f"patchtst_{'end' if pad else 'nopad'}_{'ind' if individual else 'shared'}", {"out": out})


def test_patchtst_channel_independence_and_validation():
    _seed()
    m = _backbone(revin=False).eval()
    x = torch.randn(2, 16, 3)
    perm = torch.tensor([2, 0, 1])
    torch.testing.assert_close(m(x[:, :, perm]), m(x)[:, :, perm], atol=1e-5, rtol=1e-4)
    with pytest.raises(ValueError):
        m(torch.randn(2, 15, 3))
    with pytest.raises(ValueError):
        _backbone(head_type="other")
    with pytest.raises(ValueError):
        _backbone(padding_patch="start")
    with pytest.raises(ValueError):
        _backbone(patch_len=17)


# ----------------------------------------------------------------------------- embed
def test_positional_embedding():
    pe = PositionalEmbedding(8, max_len=20)
    assert list(pe.state_dict()) == ["pe"] and pe.pe.shape == (1, 20, 8)
    out = pe(torch.zeros(2, 5, 3))
    assert out.shape == (1, 5, 8)
    assert torch.allclose(out[0, 0, 0::2], torch.zeros(4)) and torch.allclose(out[0, 0, 1::2], torch.ones(4))
    assert_reference("embed_positional", {"pe": out})


def test_token_and_feature_embeddings():
    _seed()
    t = TokenEmbedding(3, 8)
    assert list(t.state_dict()) == ["tokenConv.weight"]
    x = torch.randn(2, 10, 3, requires_grad=True)
    out = t(x)
    assert out.shape == (2, 10, 8)
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    f = TimeFeatureEmbedding(8)
    assert list(f.state_dict()) == ["embed.weight"] and f.embed.weight.shape == (8, 6)
    marks = torch.randn(2, 10, 6)
    assert f(marks).shape == (2, 10, 8)
    assert TimeFeatureEmbedding(8, input_dim=4).embed.in_features == 4
    assert_reference("embed_token", {"tok": out.detach(), "feat": f(marks).detach()})


def test_fixed_and_temporal_embedding():
    fe = FixedEmbedding(24, 8)
    assert not fe.emb.weight.requires_grad
    idx = torch.tensor([[0, 5, 23]])
    assert fe(idx).shape == (1, 3, 8) and not fe(idx).requires_grad
    _seed()
    marks = torch.stack(
        [torch.randint(0, 13, (2, 6)), torch.randint(0, 32, (2, 6)), torch.randint(0, 7, (2, 6)),
         torch.randint(0, 24, (2, 6)), torch.randint(0, 4, (2, 6))], -1).float()
    te = TemporalEmbedding(8)
    assert not hasattr(te, "minute_embed")
    out = te(marks)
    assert out.shape == (2, 6, 8)
    tm = TemporalEmbedding(8, freq="t")
    assert tm.minute_embed.emb.num_embeddings == 4
    assert not torch.allclose(tm(marks), out)
    learned = TemporalEmbedding(8, embed_type="learned")
    assert isinstance(learned.hour_embed, nn.Embedding) and learned.hour_embed.weight.requires_grad
    bad = marks.clone()
    bad[..., 3] = 24
    with pytest.raises(IndexError):
        te(bad)
    assert_reference("embed_temporal", {"fixed": out, "minute": tm(marks)})


def test_data_embeddings():
    _seed()
    x = torch.randn(2, 10, 3)
    marks6 = torch.randn(2, 10, 6)
    de = DataEmbedding(3, 8, "timeF", dropout=0.0).eval()
    out = de(x, marks6)
    assert out.shape == (2, 10, 8)
    assert de(x, None).shape == (2, 10, 8)
    assert "position_embedding.pe" in de.state_dict()
    assert DataEmbedding(3, 8, "timeF", time_feature_dim=4).temporal_embedding.embed.in_features == 4
    marks5 = torch.randint(0, 4, (2, 10, 5)).float()
    assert DataEmbedding(3, 8, "fixed", dropout=0.0).eval()(x, marks5).shape == (2, 10, 8)

    wo = DataEmbedding_wo_pos(3, 8, "timeF", dropout=0.0).eval()
    ow = wo(x, marks6)
    assert ow.shape == (2, 10, 8) and "position_embedding.pe" in wo.state_dict()
    wo.value_embedding.load_state_dict(de.value_embedding.state_dict())
    wo.temporal_embedding.load_state_dict(de.temporal_embedding.state_dict())
    torch.testing.assert_close(out - wo(x, marks6), de.position_embedding(x).expand(2, -1, -1), atol=1e-5, rtol=1e-4)

    inv = DataEmbedding_inverted(10, 8, dropout=0.0).eval()
    assert inv(x, None).shape == (2, 3, 8)
    assert inv(x, marks6).shape == (2, 9, 8)
    x.requires_grad_(True)
    de.train()
    de(x, marks6).sum().backward()
    assert torch.isfinite(x.grad).all()
    _grads_ok(de, skip=("position",))
    assert_reference("embed_data", {"de": out, "wo_pos": ow, "inv": inv(x.detach(), None).detach()})


def test_patch_embedding():
    _seed()
    pe = PatchEmbedding(8, 4, 2, 2, 0.0).eval()
    x = torch.randn(2, 3, 10, requires_grad=True)
    out, n_vars = pe(x)
    assert out.shape == (6, 5, 8) and n_vars == 3
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert {"value_embedding.weight", "position_embedding.pe"} == set(pe.state_dict())
    assert PatchEmbedding(8, 4, 4, 0, 0.0)(torch.randn(1, 2, 16))[0].shape == (2, 4, 8)
    assert_reference("embed_patch", {"out": out.detach()})

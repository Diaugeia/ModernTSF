"""Contract tests for signal/sequence components.

Covers energy_frequency_pooling, harmonic_energy_gate, haar_dwt1d, wavelet,
spectral_descriptor, hyper_state_scan, mamba, marks, quantile_head, soft_tree.
Numerical references live in ``tests/fixtures/components/<name>.pt``; regenerate
with ``TSFLAB_REGEN_COMPONENT_FIXTURES=1`` only for an intentional change.
Parameters are overwritten from a seeded generator so fixtures do not depend on
default initializers.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.component_reference import assert_reference
from tsflab.models._components.energy_frequency_pooling import EnergyBasedFrequencyPooling
from tsflab.models._components.haar_dwt1d import HaarDWT1D, HaarIDWT1D
from tsflab.models._components.harmonic_energy_gate import HarmonicEnergyGate
from tsflab.models._components.hyper_state_scan import GridStateMixer, diagonal_selective_scan
from tsflab.models._components.mamba import MambaBlock, MambaResidualBlock, RMSNorm
from tsflab.models._components.marks import (
    TIME_FEATURES,
    TSLIB_TIME_FEATURE_DIMS,
    adapt_tslib_marks,
    coerce_time_length,
    future_time_features,
    normalized_time_features,
    to_calendar_spatiotemporal,
    to_spatiotemporal,
    tslib_time_feature_dimension,
)
from tsflab.models._components.quantile_head import (
    DEFAULT_QUANTILE_LEVELS,
    QuantileHead,
    validate_quantile_levels,
)
from tsflab.models._components.soft_tree import SoftDecisionTree, SoftObliviousTree, binary_routes
from tsflab.models._components.spectral_descriptor import SpectralDescriptor
from tsflab.models._components.wavelet import (
    DecimatedWaveletTransform,
    UndecimatedWaveletTransform,
    available_wavelets,
)


def randn(*shape: int, seed: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=dtype)


def seed_params(module: torch.nn.Module, seed: int = 123, scale: float = 0.3) -> torch.nn.Module:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * scale)
    return module


def sines(batch: int = 2, length: int = 32, channels: int = 3) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32)
    base = torch.stack([torch.sin(2 * math.pi * (c + 2) * t / length) for c in range(channels)], dim=-1)
    return base.unsqueeze(0).repeat(batch, 1, 1) + 0.05 * randn(batch, length, channels, seed=7)


# ---------------------------------------------------------------- energy_frequency_pooling
def _spectrum() -> torch.Tensor:
    return torch.complex(randn(2, 4, 5, seed=1), randn(2, 4, 5, seed=2))


def test_energy_pooling_shape_and_broadcast_eval() -> None:
    m = EnergyBasedFrequencyPooling().eval()
    s = _spectrum()
    out = m(s)
    assert out.shape == s.shape and out.dtype == s.dtype and torch.is_complex(out)
    assert len(m.state_dict()) == 0
    assert torch.equal(out, out[:, :1].expand_as(out))
    # eval pick is the argmax-energy token per bin
    idx = s.abs().pow(2).argmax(dim=1)
    expect = torch.gather(s, 1, idx.unsqueeze(1))
    assert torch.equal(out[:, :1], expect)
    assert_reference("energy_frequency_pooling", {"out_real": out.real, "out_imag": out.imag})


def test_energy_pooling_train_samples_existing_tokens_and_grads() -> None:
    m = EnergyBasedFrequencyPooling().train()
    s = _spectrum().requires_grad_(True)
    torch.manual_seed(0)
    out = m(s)
    assert out.shape == s.shape
    # each pooled value equals one of the token values at that bin
    for b in range(2):
        for f in range(5):
            assert any(torch.equal(out[b, 0, f], s[b, t, f]) for t in range(4))
    out.abs().sum().backward()
    assert s.grad is not None and s.grad.abs().sum() > 0


def test_energy_pooling_rejects_bad_input() -> None:
    m = EnergyBasedFrequencyPooling()
    with pytest.raises(ValueError):
        m(torch.zeros(2, 3, 4))
    with pytest.raises(ValueError):
        m(torch.zeros(3, 4, dtype=torch.complex64))


# ---------------------------------------------------------------- harmonic_energy_gate
def test_harmonic_gate_contract() -> None:
    g = HarmonicEnergyGate(num_harmonics=3, low_freq_guard=3)
    assert len(g.state_dict()) == 0
    x = sines()
    out = g(x)
    assert out.shape == (2, 1, 3) and out.dtype == x.dtype
    assert (out >= 0).all() and (out <= 1 + 1e-6).all()
    # pure harmonic sine is near 1; white noise is lower
    noise = randn(2, 64, 3, seed=5)
    pure = torch.sin(2 * math.pi * 4 * torch.arange(64.0) / 64).view(1, 64, 1).repeat(2, 1, 3)
    assert g(pure).min() > 0.99
    assert g(noise).max() < g(pure).min()
    assert_reference("harmonic_energy_gate", {"sines": out, "noise": g(noise), "pure": g(pure)})


def test_harmonic_gate_errors() -> None:
    with pytest.raises(ValueError):
        HarmonicEnergyGate(num_harmonics=0)
    with pytest.raises(ValueError):
        HarmonicEnergyGate(low_freq_guard=0)
    with pytest.raises(ValueError):
        HarmonicEnergyGate()(torch.zeros(4, 8))
    with pytest.raises(ValueError):
        HarmonicEnergyGate(low_freq_guard=5)(torch.zeros(1, 8, 2))


def test_harmonic_gate_dtype_double_and_grad() -> None:
    g = HarmonicEnergyGate()
    x = sines().double().requires_grad_(True)
    out = g(x)
    assert out.dtype == torch.float64
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------- haar_dwt1d
@pytest.mark.parametrize("length", [8, 9])
def test_haar_round_trip_and_shapes(length: int) -> None:
    x = randn(2, 3, length, seed=3)
    a, d = HaarDWT1D()(x)
    n = math.ceil(length / 2)
    assert a.shape == d.shape == (2, 3, n)
    y = HaarIDWT1D()(a, d, length=length)
    torch.testing.assert_close(y, x, atol=1e-6, rtol=1e-6)
    assert HaarIDWT1D()(a, d).shape[-1] == 2 * n


def test_haar_energy_preserved_reference_and_grad() -> None:
    x = randn(2, 3, 8, seed=3).requires_grad_(True)
    a, d = HaarDWT1D()(x)
    torch.testing.assert_close(a.pow(2).sum() + d.pow(2).sum(), x.pow(2).sum(), atol=1e-5, rtol=1e-5)
    (a.sum() + d.pow(2).sum()).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    odd_a, odd_d = HaarDWT1D()(randn(1, 2, 7, seed=4))
    assert_reference("haar_dwt1d", {"approx": a, "detail": d, "odd_approx": odd_a, "odd_detail": odd_d})
    assert len(HaarDWT1D().state_dict()) == 0 and len(HaarIDWT1D().state_dict()) == 0


def test_haar_errors() -> None:
    with pytest.raises(ValueError):
        HaarDWT1D()(torch.zeros(1, 1, 1))
    with pytest.raises(ValueError):
        HaarIDWT1D()(torch.zeros(1, 1, 2), torch.zeros(1, 1, 3))


# ---------------------------------------------------------------- wavelet
def test_wavelet_names_and_errors() -> None:
    assert available_wavelets() == ("db1", "db2", "db4", "haar")
    with pytest.raises(ValueError):
        DecimatedWaveletTransform("nope")
    with pytest.raises(ValueError):
        DecimatedWaveletTransform("haar", level=0)
    with pytest.raises(ValueError):
        UndecimatedWaveletTransform("nope")
    with pytest.raises(ValueError):
        UndecimatedWaveletTransform("db4", level=0)


@pytest.mark.parametrize("length", [16, 15])
def test_decimated_haar_round_trip(length: int) -> None:
    x = randn(2, 3, length, seed=11)
    t = DecimatedWaveletTransform("haar", level=2)
    coeffs = t.decompose(x)
    assert len(coeffs) == 3
    torch.testing.assert_close(t.reconstruct(coeffs), x, atol=1e-5, rtol=1e-5)
    assert set(t.state_dict()) == {"low", "high"}


def test_decimated_shapes_db4_and_no_reconstruct() -> None:
    x = randn(2, 3, 16, seed=11)
    t = DecimatedWaveletTransform("db4", level=2)
    coeffs = t.decompose(x)
    assert [c.shape for c in coeffs] == [(2, 3, 9), (2, 3, 9), (2, 3, 11)]  # circular padding of filter_len - 2 per side lengthens long filters
    with pytest.raises(NotImplementedError):
        t.reconstruct(coeffs)
    with pytest.raises(ValueError):
        DecimatedWaveletTransform("haar", level=2).reconstruct([x])


def test_wavelet_filters_orthonormal() -> None:
    for name in ("haar", "db2", "db4"):
        t = DecimatedWaveletTransform(name)
        low, high = t.low.flatten(), t.high.flatten()
        assert abs(float(low.pow(2).sum()) - 1) < 1e-5
        assert abs(float(high.pow(2).sum()) - 1) < 1e-5
        assert abs(float((low * high).sum())) < 1e-5


def test_undecimated_short_input_raises() -> None:
    with pytest.raises(RuntimeError):
        UndecimatedWaveletTransform("db4", level=3)(randn(1, 1, 16))


def test_undecimated_shapes_reference_and_grad() -> None:
    # circular padding needs length >= (filter_len - 1) * 2**(level - 1) = 28
    x = randn(2, 3, 32, seed=12).requires_grad_(True)
    u = UndecimatedWaveletTransform("db4", level=3)
    coeffs = u(x)
    assert len(coeffs) == 4 and all(c.shape == x.shape and c.dtype == x.dtype for c in coeffs)
    sum(c.pow(2).sum() for c in coeffs).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    dec = DecimatedWaveletTransform("db2", level=2).decompose(x.detach())
    ref = {f"swt{i}": c for i, c in enumerate(coeffs)}
    ref.update({f"dwt{i}": c for i, c in enumerate(dec)})
    assert_reference("wavelet", ref)


# ---------------------------------------------------------------- spectral_descriptor
def test_spectral_descriptor_contract() -> None:
    sd = SpectralDescriptor()
    assert len(sd.state_dict()) == 0
    x = sines(length=33)
    out = sd(x)
    assert out.shape == (2, 4) and out.dtype == x.dtype
    assert (out[:, 0] >= 0).all() and (out[:, 0] <= 1 + 1e-6).all()
    torch.testing.assert_close(out[:, 1:].sum(dim=1), torch.ones(2), atol=1e-5, rtol=1e-5)
    # constant input: flat (smoothed) spectrum, entropy ~ 1
    flat = sd(torch.ones(1, 32, 2))
    assert flat[0, 0] > 0.99
    # low-frequency tone concentrates in the low band, high-frequency tone in the high band
    t = torch.arange(48.0)
    low = torch.sin(2 * math.pi * 1 * t / 48).view(1, 48, 1)
    high = torch.sin(2 * math.pi * 22 * t / 48).view(1, 48, 1)
    assert sd(low)[0, 1] > 0.99 and sd(high)[0, 3] > 0.99
    assert_reference("spectral_descriptor", {"sines": out, "low": sd(low), "high": sd(high)})


def test_spectral_descriptor_errors_and_grad() -> None:
    with pytest.raises(ValueError):
        SpectralDescriptor(eps=0.0)
    with pytest.raises(ValueError):
        SpectralDescriptor()(torch.zeros(4, 8))
    x = sines().requires_grad_(True)
    SpectralDescriptor()(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------- hyper_state_scan
def test_diagonal_scan_recurrence_and_shape() -> None:
    u, delta, b = randn(2, 3, 6, seed=1), randn(2, 3, 6, seed=2).abs(), randn(2, 3, 6, seed=3)
    a = -randn(3, seed=4).abs()
    h = diagonal_selective_scan(u, delta, a, b)
    assert h.shape == u.shape and h.dtype == u.dtype
    state = torch.zeros(2, 3)
    for t in range(6):
        state = torch.exp(delta[..., t] * a) * state + delta[..., t] * b[..., t] * u[..., t]
        torch.testing.assert_close(h[..., t], state, atol=1e-6, rtol=1e-6)
    # zero input gives zero state
    assert diagonal_selective_scan(torch.zeros_like(u), delta, a, b).abs().sum() == 0
    assert_reference("hyper_state_scan", {"scan": h})


def test_diagonal_scan_errors_and_grad() -> None:
    u = randn(1, 2, 4, seed=1).requires_grad_(True)
    a = (-torch.ones(2)).requires_grad_(True)
    with pytest.raises(ValueError):
        diagonal_selective_scan(u, u, a, torch.zeros(1, 2, 5))
    with pytest.raises(ValueError):
        diagonal_selective_scan(u, u, torch.ones(3), u)
    diagonal_selective_scan(u, u.abs(), a, u).sum().backward()
    assert u.grad is not None and a.grad is not None and a.grad.abs().sum() > 0


def test_grid_state_mixer() -> None:
    m = seed_params(GridStateMixer(4, kernel_size=3))
    assert set(m.state_dict()) == {"conv.weight", "conv.bias"}
    assert m.conv.groups == 4
    x = randn(2, 4, 5, 6, seed=9).requires_grad_(True)
    out = m(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    # depthwise: channel 0 output depends only on channel 0 input
    x2 = x.detach().clone()
    x2[:, 1:] += 1.0
    torch.testing.assert_close(m(x2)[:, 0], out[:, 0].detach())
    out.sum().backward()
    assert x.grad is not None and m.conv.weight.grad is not None
    for bad in ((0, 3), (2, 2)):
        with pytest.raises(ValueError):
            GridStateMixer(*bad)
    with pytest.raises(ValueError):
        m(torch.zeros(2, 4, 5))


def test_grid_state_mixer_reference() -> None:
    m = seed_params(GridStateMixer(3, kernel_size=3))
    assert_reference("hyper_state_scan_mixer", {"out": m(randn(1, 3, 4, 5, seed=2))})


# ---------------------------------------------------------------- mamba
def _mamba_block() -> MambaBlock:
    return seed_params(MambaBlock(d_model=6, d_inner=8, dt_rank=2, d_conv=3, d_state=4), scale=0.2)


def test_mamba_state_dict_keys() -> None:
    block = _mamba_block()
    assert set(block.state_dict()) == {
        "A_log", "D", "in_proj.weight", "conv1d.weight", "conv1d.bias",
        "x_proj.weight", "dt_proj.weight", "dt_proj.bias", "out_proj.weight",
    }
    assert set(RMSNorm(6).state_dict()) == {"weight"}
    res = MambaResidualBlock(6, 8, 2, 3, 4)
    assert {k.split(".")[0] for k in res.state_dict()} == {"mixer", "norm"}


def test_mamba_default_init() -> None:
    block = MambaBlock(6, 8, 2, 3, 4)
    expected = torch.log(torch.arange(1, 5).float()).repeat(8, 1)
    torch.testing.assert_close(block.A_log, expected)
    assert torch.equal(block.D, torch.ones(8))


def test_mamba_options_keep_default_keys_and_extend_cleanly() -> None:
    default = MambaBlock(6, 8, 2, 3, 4)
    optioned = MambaBlock(6, 8, 2, 3, 4, x_dropout=0.3, reference_dt_init=True)
    assert set(default.state_dict()) == set(optioned.state_dict())
    no_conv = MambaBlock(6, 8, 2, 3, 4, use_conv=False)
    assert not any(k.startswith("conv1d") for k in no_conv.state_dict())
    x = randn(2, 5, 6, seed=3)
    assert no_conv(x).shape == x.shape
    with pytest.raises(ValueError):
        MambaBlock(6, 8, 2, 3, 4, x_dropout=1.0)


def test_mamba_reference_dt_init_bias_inverts_to_step_range() -> None:
    block = MambaBlock(6, 64, 2, 3, 4, reference_dt_init=True)
    step = torch.nn.functional.softplus(block.dt_proj.bias)
    assert step.min() >= 1e-4 - 1e-7 and step.max() <= 0.1 + 1e-6
    assert block.dt_proj.weight.abs().max() <= 2 ** -0.5 + 1e-6


def test_mamba_x_dropout_only_acts_in_training() -> None:
    block = seed_params(MambaBlock(6, 8, 2, 3, 4, x_dropout=0.5), scale=0.2)
    x = randn(2, 7, 6, seed=2)
    block.eval()
    torch.testing.assert_close(block(x), block(x))
    block.train()
    torch.manual_seed(0)
    first = block(x)
    torch.manual_seed(1)
    assert not torch.allclose(first, block(x))


def test_rmsnorm_contract() -> None:
    n = RMSNorm(6)
    x = randn(2, 5, 6, seed=1) * 3
    y = n(x)
    assert y.shape == x.shape and y.dtype == x.dtype
    torch.testing.assert_close(y.pow(2).mean(-1), torch.ones(2, 5), atol=1e-3, rtol=1e-3)


def test_mamba_block_shape_causality_grad_reference() -> None:
    block = _mamba_block()
    x = randn(2, 7, 6, seed=2).requires_grad_(True)
    out = block(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    # causal: changing future steps does not alter earlier outputs
    x2 = x.detach().clone()
    x2[:, 4:] += 1.0
    torch.testing.assert_close(block(x2)[:, :4], out[:, :4].detach(), atol=1e-6, rtol=1e-5)
    out.pow(2).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
    res = seed_params(MambaResidualBlock(6, 8, 2, 3, 4), scale=0.2)
    rout = res(x.detach())
    assert rout.shape == x.shape
    assert_reference("mamba", {"block": out, "residual": rout, "ssm": block.ssm(randn(2, 7, 8, seed=4))})


def test_mamba_selective_scan_static_matches_loop() -> None:
    u = randn(1, 5, 3, seed=1)
    delta = randn(1, 5, 3, seed=2).abs()
    a = -randn(3, 2, seed=3).abs()
    b, c = randn(1, 5, 2, seed=4), randn(1, 5, 2, seed=5)
    d = randn(3, seed=6)
    out = MambaBlock.selective_scan(u, delta, a, b, c, d)
    assert out.shape == u.shape
    h = torch.zeros(1, 3, 2)
    for t in range(5):
        h = torch.exp(delta[:, t].unsqueeze(-1) * a) * h + delta[:, t].unsqueeze(-1) * b[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
        y = (h * c[:, t].unsqueeze(1)).sum(-1) + u[:, t] * d
        torch.testing.assert_close(out[:, t], y, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------- marks
def _raw_marks(batch: int = 2, steps: int = 4) -> torch.Tensor:
    base = torch.tensor([2020.0, 2.0, 29.0, 5.0, 13.0, 30.0])
    marks = base.repeat(batch, steps, 1)
    marks[..., 4] = torch.arange(steps, dtype=torch.float32) * 6
    marks[..., 5] = torch.arange(steps, dtype=torch.float32) * 15
    return marks


def test_marks_constants_and_dimension() -> None:
    assert TIME_FEATURES == 2
    for freq, dim in TSLIB_TIME_FEATURE_DIMS.items():
        assert tslib_time_feature_dimension(freq.upper()) == dim
    with pytest.raises(ValueError):
        tslib_time_feature_dimension("zz")


def test_normalized_time_features() -> None:
    m = _raw_marks()
    f = normalized_time_features(m)
    assert f.shape == (2, 4, 2) and f.dtype == m.dtype
    assert (f >= 0).all() and (f < 1).all()
    torch.testing.assert_close(f[0, 1, 0], torch.tensor((6 * 60 + 15) / 1440.0))
    torch.testing.assert_close(f[0, 0, 1], torch.tensor(5 / 7))


def test_to_spatiotemporal_variants() -> None:
    v = randn(2, 4, 3, seed=1)
    out = to_spatiotemporal(v, _raw_marks())
    assert out.shape == (2, 4, 3, 3) and torch.equal(out[..., 0], v)
    assert torch.equal(out[:, :, 0, 1:], out[:, :, 2, 1:])
    cov = randn(2, 4, 3, 5, seed=2)
    out4 = to_spatiotemporal(v, cov)
    assert out4.shape == (2, 4, 3, 6) and torch.equal(out4[..., 1:], cov)
    outn = to_spatiotemporal(v, None)
    assert outn.shape == (2, 4, 3, 3) and (outn[..., 1:] == 0).all()
    cal = to_calendar_spatiotemporal(v, _raw_marks())
    assert torch.equal(cal, out)
    ok = to_calendar_spatiotemporal(v, randn(2, 4, 3, 2, seed=3))
    assert ok.shape == (2, 4, 3, 3)
    with pytest.raises(ValueError):
        to_calendar_spatiotemporal(v, cov)


def test_future_time_features_and_coerce() -> None:
    m = _raw_marks()
    f = future_time_features(m, 3)
    assert f.shape == (2, 4, 3, 2)
    cov = randn(2, 4, 3, 5, seed=2)
    assert future_time_features(cov, 3) is cov
    assert coerce_time_length(m, 4) is m
    assert torch.equal(coerce_time_length(m, 2), m[:, -2:])
    padded = coerce_time_length(m, 6)
    assert padded.shape == (2, 6, 6) and torch.equal(padded[:, 4], m[:, -1]) and torch.equal(padded[:, 5], m[:, -1])
    p4 = coerce_time_length(cov, 6)
    assert p4.shape == (2, 6, 3, 5)


def test_adapt_tslib_marks() -> None:
    m = _raw_marks()
    assert adapt_tslib_marks(None, embed_type="timeF", freq="h") is None
    cat = adapt_tslib_marks(m, embed_type="fixed", freq="h")
    assert cat.shape == (2, 4, 5) and torch.equal(cat, m[..., 1:])
    five = m[..., 1:]
    assert adapt_tslib_marks(five, embed_type="learned", freq="h") is five
    tf = adapt_tslib_marks(m, embed_type="timeF", freq="h")
    assert tf.shape == (2, 4, 4)
    assert (tf >= -0.5 - 1e-6).all() and (tf <= 0.5 + 1e-6).all()
    # leap-year day of year: Feb 29 -> 31 + 29 = 60 -> (60 - 1)/365 - 0.5
    torch.testing.assert_close(tf[0, 0, 3], torch.tensor(59 / 365 - 0.5))
    pre = randn(2, 4, 4, seed=1)
    assert adapt_tslib_marks(pre, embed_type="timeF", freq="h") is pre
    with pytest.raises(ValueError):
        adapt_tslib_marks(m, embed_type="timeF", freq="t")
    with pytest.raises(ValueError):
        adapt_tslib_marks(m[0], embed_type="timeF", freq="h")
    with pytest.raises(ValueError):
        adapt_tslib_marks(m[..., :3], embed_type="fixed", freq="h")
    v = randn(2, 4, 3, seed=1)
    assert_reference(
        "marks",
        {
            "normalized": normalized_time_features(m),
            "spatiotemporal": to_spatiotemporal(v, m),
            "tslib_hourly": tf,
        },
    )


# ---------------------------------------------------------------- quantile_head
def test_quantile_levels_validation() -> None:
    assert validate_quantile_levels(None) == list(DEFAULT_QUANTILE_LEVELS)
    assert validate_quantile_levels((0.2, 0.8)) == [0.2, 0.8]
    for bad in ([], [0.0, 0.5], [0.5, 1.0], [0.5, 0.5], [0.7, 0.3]):
        with pytest.raises(ValueError):
            validate_quantile_levels(bad)


@pytest.mark.parametrize("levels", [[0.1, 0.25, 0.5, 0.75, 0.9], [0.1, 0.4, 0.7], [0.5], [0.3, 0.9]])
def test_quantile_head_monotone_and_median_anchor(levels: list[float]) -> None:
    head = seed_params(QuantileHead(levels, in_features=4), scale=0.5)
    base = randn(2, 5, 3, 4, seed=3)
    out = head(base)
    assert out.shape == (2, 5, 3, len(levels)) and out.dtype == base.dtype
    assert (out[..., 1:] >= out[..., :-1]).all()
    anchor = head.anchor_proj(base).squeeze(-1)
    torch.testing.assert_close(out[..., head.median_idx], anchor)
    expected_median = min(range(len(levels)), key=lambda i: abs(levels[i] - 0.5))
    assert head.median_idx == expected_median
    assert "_levels" not in head.state_dict()


def test_quantile_head_grad_and_reference() -> None:
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    head = seed_params(QuantileHead(levels, in_features=4), scale=0.5)
    base = randn(2, 3, 2, 4, seed=3).requires_grad_(True)
    out = head(base)
    out.sum().backward()
    assert base.grad is not None and base.grad.abs().sum() > 0
    assert all(p.grad is not None for p in head.parameters())
    assert set(head.state_dict()) == {"anchor_proj.weight", "anchor_proj.bias", "offset_proj.weight", "offset_proj.bias"}
    assert_reference("quantile_head", {"out": out})


# ---------------------------------------------------------------- soft_tree
def test_binary_routes() -> None:
    nodes, right = binary_routes(2)
    assert nodes.tolist() == [[0, 1], [0, 1], [0, 2], [0, 2]]
    assert right.tolist() == [[False, False], [False, True], [True, False], [True, True]]
    assert nodes.dtype == torch.long and right.dtype == torch.bool
    with pytest.raises(ValueError):
        binary_routes(0)


@pytest.mark.parametrize("cls", [SoftDecisionTree, SoftObliviousTree])
def test_soft_tree_probabilities_and_grad(cls) -> None:
    tree = seed_params(cls(5, 3, depth=3, temperature=0.7), scale=0.5)
    x = randn(4, 5, seed=1).requires_grad_(True)
    probs = tree.leaf_probabilities(x)
    assert probs.shape == (4, 8)
    assert (probs >= 0).all()
    torch.testing.assert_close(probs.sum(-1), torch.ones(4), atol=1e-5, rtol=1e-5)
    out = tree(x)
    assert out.shape == (4, 3) and out.dtype == x.dtype
    out.pow(2).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert all(p.grad is not None for p in tree.parameters())
    with pytest.raises(ValueError):
        tree(torch.zeros(4, 6))
    with pytest.raises(ValueError):
        tree(torch.zeros(5))
    with pytest.raises(ValueError):
        cls(0, 1)
    with pytest.raises(ValueError):
        cls(2, 1, temperature=0.0)


def test_soft_tree_default_state_and_reference() -> None:
    dt = SoftDecisionTree(5, 3, depth=2)
    assert set(dt.state_dict()) == {"split_mask", "split_weight", "threshold", "leaf_value", "route_nodes", "route_right"}
    ot = SoftObliviousTree(5, 3, depth=2)
    assert set(ot.state_dict()) == {"split_weight", "threshold", "leaf_value", "route_right"}
    assert ot.route_right.tolist() == binary_routes(2)[1].tolist()
    dt = seed_params(SoftDecisionTree(5, 3, depth=3, temperature=0.7), scale=0.5)
    ot = seed_params(SoftObliviousTree(5, 3, depth=3, temperature=0.7), scale=0.5)
    x = randn(4, 5, seed=1)
    assert_reference(
        "soft_tree",
        {"decision": dt(x), "oblivious": ot(x), "decision_probs": dt.leaf_probabilities(x)},
    )


def test_soft_decision_tree_mask_and_fixed_split() -> None:
    mask = torch.zeros(3, 4)
    mask[:, 0] = 1.0
    tree = seed_params(SoftDecisionTree(4, 2, depth=2, split_mask=mask))
    x = randn(3, 4, seed=1)
    x2 = x.clone()
    x2[:, 1:] += 5.0
    torch.testing.assert_close(tree(x), tree(x2))
    with pytest.raises(ValueError):
        SoftDecisionTree(4, 2, depth=2, split_mask=torch.zeros(3, 4))
    with pytest.raises(ValueError):
        SoftDecisionTree(4, 2, depth=2, split_mask=torch.ones(2, 4))
    w, t = randn(3, 4, seed=5), randn(3, seed=6)
    fixed = SoftDecisionTree(4, 2, depth=2, fixed_split_weight=w, fixed_threshold=t)
    names = {n for n, _ in fixed.named_parameters()}
    assert names == {"leaf_value"}
    assert {"split_weight", "threshold"} <= set(fixed.state_dict())
    with pytest.raises(ValueError):
        SoftDecisionTree(4, 2, depth=2, fixed_split_weight=w)
    with pytest.raises(ValueError):
        SoftDecisionTree(4, 2, depth=2, fixed_split_weight=w[:2], fixed_threshold=t)
    with pytest.raises(ValueError):
        SoftDecisionTree(4, 2, depth=2, fixed_split_weight=w, fixed_threshold=t[:2])

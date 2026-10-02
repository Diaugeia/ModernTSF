"""Contract tests for the basic shared components (group A).

Each test checks the card's Interface (shapes, dtype, state-dict keys, errors),
documented invariants, gradient flow, and a seeded numerical regression against
``tests/fixtures/components/<name>.pt``. Regenerate fixtures only for an
intentional numerical change (``TSFLAB_REGEN_COMPONENT_FIXTURES=1``).
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.component_reference import assert_reference
from tsflab.models._components.adain_style_norm import AdaptiveInstanceNorm1d
from tsflab.models._components.channel_alignment import fit_channels
from tsflab.models._components.channel_wise_linear import ChannelWiseLinear
from tsflab.models._components.dlinear import DLinearBackbone
from tsflab.models._components.dominant_periods import dominant_periods
from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.forecast_embedding import ForecastEmbedding, RawCalendarEmbedding
from tsflab.models._components.gated_fusion import GatedFusion
from tsflab.models._components.gaussian_parameter_head import GaussianParameterHead
from tsflab.models._components.last_value_center import center_on_last_value, restore_last_value
from tsflab.models._components.masking import LocalMask, ProbMask, TriangularCausalMask
from tsflab.models._components.mixer_block import MixerBlock
from tsflab.models._components.periodic_query_bank import PeriodicQueryBank
from tsflab.models._components.positional_encoding import positional_encoding
from tsflab.models._components.revin import RevIN
from tsflab.models._components.series_decomposition import EdgePaddedMovingAverage, SeriesDecomposition


def _rand(*shape: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator)


def _grads(module: torch.nn.Module) -> list[torch.Tensor | None]:
    return [p.grad for p in module.parameters()]


# ---------------------------------------------------------------- adain_style_norm
def test_adain_contract_and_reference() -> None:
    module = AdaptiveInstanceNorm1d()
    assert module.state_dict() == {}
    x = _rand(2, 8, 3).requires_grad_(True)
    mean, std = _rand(2, 1, 3, seed=1).requires_grad_(True), (_rand(2, 1, 3, seed=2).abs() + 0.5).requires_grad_(True)
    out = module(x, mean, std)
    assert out.shape == x.shape and out.dtype == x.dtype
    # unit-normalized then rescaled: output stats equal the style stats (up to eps)
    torch.testing.assert_close(out.mean(1, keepdim=True), mean, atol=1e-4, rtol=0)
    torch.testing.assert_close(out.std(1, keepdim=True), std, atol=1e-3, rtol=1e-3)
    out.square().sum().backward()
    assert x.grad is not None and mean.grad is not None and std.grad is not None
    # scalar style broadcasts
    assert module(x, torch.zeros(1), torch.ones(1)).shape == x.shape
    with pytest.raises(ValueError):
        module(_rand(4, 3), mean, std)
    assert_reference("adain_style_norm", {"out": out, "x_grad": x.grad})


# ---------------------------------------------------------------- channel_alignment
def test_fit_channels_contract_and_reference() -> None:
    x = _rand(2, 4, 3).requires_grad_(True)
    padded = fit_channels(x, 5)
    assert padded.shape == (2, 4, 5)
    assert torch.equal(padded[..., :3], x) and not padded[..., 3:].any()
    sliced = fit_channels(x, 2)
    assert sliced.shape == (2, 4, 2) and torch.equal(sliced, x[..., :2])
    same = fit_channels(x, 3)
    assert same is not x and torch.equal(same, x)
    (padded.sum() + sliced.sum()).backward()
    assert torch.equal(x.grad, torch.tensor([2.0, 2.0, 1.0]).expand_as(x))
    assert fit_channels(torch.ones(3, dtype=torch.float64), 4).dtype == torch.float64
    assert fit_channels(torch.ones(2, dtype=torch.int64), 3).dtype == torch.int64
    with pytest.raises(ValueError, match="width must be positive"):
        fit_channels(x, 0)
    assert_reference("channel_alignment", {"padded": padded, "sliced": sliced})


# ---------------------------------------------------------------- channel_wise_linear
@pytest.mark.parametrize("individual", [False, True])
def test_channel_wise_linear_contract_and_reference(individual: bool) -> None:
    torch.manual_seed(0)
    module = ChannelWiseLinear(6, 4, 3, individual=individual)
    keys = set(module.state_dict())
    if individual:
        assert keys == {f"linears.{c}.{p}" for c in range(3) for p in ("weight", "bias")}
    else:
        assert keys == {"linear.weight", "linear.bias"}
        assert module.linear.weight.shape == (4, 6)
    x = _rand(2, 3, 6).requires_grad_(True)
    out = module(x)
    assert out.shape == (2, 3, 4) and out.dtype == x.dtype
    out.square().sum().backward()
    assert x.grad is not None and all(g is not None and g.abs().sum() > 0 for g in _grads(module))
    with pytest.raises(ValueError):
        module(_rand(3, 6))
    with pytest.raises(ValueError):
        module(_rand(2, 4, 6))
    if individual:  # channels are independent
        x2 = x.detach().clone()
        x2[:, 0] += 1.0
        assert torch.equal(module(x2)[:, 1:], out.detach()[:, 1:])
    assert_reference(f"channel_wise_linear_{'ind' if individual else 'shared'}", {"out": out, "x_grad": x.grad})


# ---------------------------------------------------------------- dlinear
@pytest.mark.parametrize("individual", [False, True])
def test_dlinear_contract_and_reference(individual: bool) -> None:
    torch.manual_seed(0)
    module = DLinearBackbone(3, 12, 5, kernel_size=5, individual=individual)
    keys = set(module.state_dict())
    stem = "linears.0" if individual else "linear"
    assert f"seasonal_projection.{stem}.weight" in keys and f"trend_projection.{stem}.bias" in keys
    assert not any(k.startswith("decomposition") for k in keys)
    x = _rand(2, 12, 3).requires_grad_(True)
    out = module(x)
    assert out.shape == (2, 5, 3) and out.dtype == x.dtype
    # equals the sum of the two projections of the decomposition
    seasonal, trend = module.decomposition(x)
    manual = module.seasonal_projection(seasonal.permute(0, 2, 1)) + module.trend_projection(trend.permute(0, 2, 1))
    torch.testing.assert_close(out, manual.permute(0, 2, 1))
    out.square().sum().backward()
    assert x.grad is not None and all(g is not None for g in _grads(module))
    with pytest.raises(ValueError):
        module(_rand(2, 11, 3))
    with pytest.raises(ValueError):
        DLinearBackbone(3, 12, 5, kernel_size=4)
    assert DLinearBackbone(3, 6, 2, kernel_size=25)(_rand(1, 6, 3)).shape == (1, 2, 3)
    assert_reference(f"dlinear_{'ind' if individual else 'shared'}", {"out": out, "x_grad": x.grad})


# ---------------------------------------------------------------- dominant_periods
def test_dominant_periods_contract_and_reference() -> None:
    t = torch.arange(24, dtype=torch.float32)
    x = (torch.sin(2 * math.pi * t / 6) + 0.3 * torch.sin(2 * math.pi * t / 12 * 1.0))[None, :, None].repeat(2, 1, 2)
    x = (x + 0.01 * _rand(2, 24, 2)).requires_grad_(True)
    periods, amps = dominant_periods(x, k=2)
    assert periods.shape == (2,) and periods.dtype.kind == "i"
    assert sorted(periods.tolist()) == [6, 12] and periods[0] == 6
    assert amps.shape == (2, 2) and amps.dtype == x.dtype
    amps.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    p1, a1 = dominant_periods(x.detach(), k=1)
    assert p1.shape == (1,) and a1.shape == (2, 1)
    for bad in (0, 13):
        with pytest.raises(ValueError):
            dominant_periods(x, k=bad)
    with pytest.raises(ValueError):
        dominant_periods(_rand(24, 2))
    assert_reference("dominant_periods", {"periods": torch.as_tensor(periods), "amps": amps})


# ---------------------------------------------------------------- last_value_center
def test_last_value_center_contract_and_reference() -> None:
    x = _rand(2, 5, 3).requires_grad_(True)
    centered, level = center_on_last_value(x)
    assert centered.shape == x.shape and level.shape == (2, 1, 3) and not level.requires_grad
    assert not centered[:, -1].any()
    head = _rand(2, 4, 3, seed=3)
    restored = restore_last_value(head, level)
    assert restored.shape == (2, 4, 3)
    torch.testing.assert_close(restore_last_value(centered, level), x.detach())
    assert torch.equal(restore_last_value(head, 0.0), head)
    (centered.sum() + restored.sum()).backward()
    # level is detached: gradient only through the centered term (1 everywhere)
    assert torch.equal(x.grad, torch.ones_like(x))
    assert_reference("last_value_center", {"centered": centered, "restored": restored})


# ---------------------------------------------------------------- revin
@pytest.mark.parametrize("affine,subtract_last", [(True, False), (False, False), (True, True)])
def test_revin_contract_and_reference(affine: bool, subtract_last: bool) -> None:
    torch.manual_seed(0)
    module = RevIN(3, affine=affine, subtract_last=subtract_last)
    if affine:
        with torch.no_grad():
            module.affine_weight.copy_(torch.tensor([0.5, 1.5, 2.0]))
            module.affine_bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
    assert set(module.state_dict()) == ({"affine_weight", "affine_bias"} if affine else set())
    x = (_rand(2, 8, 3) * 3 + 5).requires_grad_(True)
    y = module(x, "norm")
    assert y.shape == x.shape and y.dtype == x.dtype
    if not affine and not subtract_last:
        torch.testing.assert_close(y.mean(1), torch.zeros(2, 3), atol=1e-5, rtol=0)
        torch.testing.assert_close(y.var(1, unbiased=False), torch.ones(2, 3), atol=1e-3, rtol=0)
    if subtract_last and not affine:
        assert torch.allclose(y[:, -1], torch.zeros(2, 3))
    torch.testing.assert_close(module(y, "denorm"), x, atol=1e-4, rtol=1e-4)
    horizon = module(_rand(2, 4, 3, seed=5), "denorm")  # different time length
    assert horizon.shape == (2, 4, 3)
    y.square().sum().backward()
    assert x.grad is not None
    if affine:
        assert module.affine_weight.grad is not None and module.affine_bias.grad is not None
    # rank > 3 reduces over all middle axes
    assert module(_rand(2, 4, 5, 3), "norm").shape == (2, 4, 5, 3)
    assert_reference(f"revin_{int(affine)}{int(subtract_last)}", {"y": y, "horizon": horizon, "x_grad": x.grad})


def test_revin_errors_and_disabled() -> None:
    module = RevIN(3)
    with pytest.raises(RuntimeError):
        module(_rand(2, 4, 3), "denorm")
    with pytest.raises(ValueError):
        module(_rand(4, 3), "norm")
    with pytest.raises(ValueError):
        module(_rand(2, 4, 2), "norm")
    with pytest.raises(ValueError):
        module(_rand(2, 4, 3), "bogus")
    x = _rand(2, 4, 3)
    off = RevIN(3, enabled=False)
    assert off(x, "norm") is x and off(x, "denorm") is x
    with pytest.raises(ValueError):
        RevIN(0)
    with pytest.raises(ValueError):
        RevIN(3, eps=0.0)


# ---------------------------------------------------------------- series_decomposition
def test_series_decomposition_contract_and_reference() -> None:
    module = SeriesDecomposition(5)
    assert module.state_dict() == {} and EdgePaddedMovingAverage(3).state_dict() == {}
    x = _rand(2, 10, 3).requires_grad_(True)
    residual, trend = module(x)
    assert residual.shape == trend.shape == x.shape and residual.dtype == x.dtype
    torch.testing.assert_close(residual + trend, x)
    # hand-computed edge-padded moving average at the first step
    expected = (x[:, 0] * 3 + x[:, 1] + x[:, 2]) / 5
    torch.testing.assert_close(trend[:, 0], expected)
    ones = torch.ones(1, 6, 2)
    torch.testing.assert_close(module(ones)[1], ones)
    (residual.square().sum() + trend.sum()).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert SeriesDecomposition(31)(x)[1].shape == x.shape  # kernel > length
    assert EdgePaddedMovingAverage(3, stride=2)(x).shape == (2, 5, 3)
    assert EdgePaddedMovingAverage(1)(x).shape == x.shape
    for bad in (0, 4, -3):
        with pytest.raises(ValueError):
            EdgePaddedMovingAverage(bad)
    with pytest.raises(ValueError):
        EdgePaddedMovingAverage(3, stride=0)
    assert_reference("series_decomposition", {"residual": residual, "trend": trend, "x_grad": x.grad})


# ---------------------------------------------------------------- flatten_forecast_head
@pytest.mark.parametrize("individual", [False, True])
def test_flatten_forecast_head_contract_and_reference(individual: bool) -> None:
    torch.manual_seed(0)
    module = FlattenForecastHead(individual, 3, 8, 5)
    keys = set(module.state_dict())
    if individual:
        assert keys == {f"linears.{i}.{p}" for i in range(3) for p in ("weight", "bias")}
    else:
        assert keys == {"linear.weight", "linear.bias"}
    x = _rand(2, 3, 4, 2).requires_grad_(True)
    out = module(x)
    assert out.shape == (2, 3, 5) and out.dtype == x.dtype
    out.square().sum().backward()
    assert x.grad is not None and all(g is not None for g in _grads(module))
    if not individual:
        assert module(_rand(7, 2, 3, 4, 2)).shape == (7, 2, 3, 5)  # leading axes preserved
    with pytest.raises(RuntimeError):
        module(_rand(2, 3, 4, 3))
    assert_reference(f"flatten_forecast_head_{'ind' if individual else 'shared'}", {"out": out, "x_grad": x.grad})


def test_flatten_forecast_head_dropout_eval_deterministic() -> None:
    module = FlattenForecastHead(False, 3, 8, 5, head_dropout=0.5).eval()
    x = _rand(2, 3, 4, 2)
    assert torch.equal(module(x), module(x))


# ---------------------------------------------------------------- forecast_embedding
def test_forecast_embedding_contract_and_reference() -> None:
    torch.manual_seed(0)
    calendar = RawCalendarEmbedding(6)
    assert set(calendar.state_dict()) == {"projection.weight"}
    marks = torch.tensor([[[2024.0, 3.0, 15.0, 2.0, 12.0, 30.0], [2024.0, 3.0, 15.0, 2.0, 13.0, 30.0]]]).repeat(2, 1, 1)
    cal = calendar(marks)
    assert cal.shape == (2, 2, 6) and cal.dtype == marks.dtype
    torch.testing.assert_close(cal, calendar.projection(marks / torch.tensor([2100.0, 12.0, 31.0, 6.0, 23.0, 59.0]) - 0.5))
    with pytest.raises(ValueError):
        calendar(_rand(2, 2, 5))
    with pytest.raises(ValueError):
        calendar(_rand(2, 6))

    torch.manual_seed(0)
    module = ForecastEmbedding(3, 6, 0.0)
    assert set(module.state_dict()) == {"value.weight", "value.bias", "calendar.projection.weight"}
    values = _rand(2, 2, 3).requires_grad_(True)
    out = module(values, marks)
    assert out.shape == (2, 2, 6) and out.dtype == values.dtype
    out.square().sum().backward()
    assert values.grad is not None and all(g is not None for g in _grads(module))
    with pytest.raises(ValueError):
        module(_rand(2, 3), marks)
    with pytest.raises(ValueError):
        module(values, marks[:, :1])
    assert_reference("forecast_embedding", {"calendar": cal, "out": out, "x_grad": values.grad})


# ---------------------------------------------------------------- gated_fusion
def test_gated_fusion_contract_and_reference() -> None:
    torch.manual_seed(0)
    module = GatedFusion(4)
    assert set(module.state_dict()) == {"gate_a.weight", "gate_a.bias", "gate_b.weight", "gate_b.bias"}
    a, b = _rand(2, 3, 4).requires_grad_(True), _rand(2, 3, 4, seed=1).requires_grad_(True)
    out = module(a, b)
    assert out.shape == a.shape and out.dtype == a.dtype
    # convex blend: each output element lies between the two inputs
    lo, hi = torch.minimum(a, b).detach(), torch.maximum(a, b).detach()
    assert (out.detach() >= lo - 1e-6).all() and (out.detach() <= hi + 1e-6).all()
    out.square().sum().backward()
    assert a.grad is not None and b.grad is not None and all(g is not None for g in _grads(module))
    assert module(_rand(5, 4), _rand(5, 4, seed=1)).shape == (5, 4)
    with pytest.raises(ValueError):
        module(a, _rand(2, 4, 4))
    with pytest.raises(ValueError):
        GatedFusion(0)
    assert_reference("gated_fusion", {"out": out, "a_grad": a.grad, "b_grad": b.grad})


# ---------------------------------------------------------------- gaussian_parameter_head
@pytest.mark.parametrize("transform", ["softplus", "log1pexp"])
def test_gaussian_parameter_head_contract_and_reference(transform: str) -> None:
    torch.manual_seed(0)
    module = GaussianParameterHead(4, 3, scale_transform=transform)
    assert set(module.state_dict()) == {"loc_layer.weight", "loc_layer.bias", "scale_layer.weight", "scale_layer.bias"}
    x = _rand(2, 5, 4).requires_grad_(True)
    loc, scale = module(x)
    assert loc.shape == scale.shape == (2, 5, 3) and loc.dtype == x.dtype
    assert (scale >= module.eps).all()
    torch.testing.assert_close(scale, torch.nn.functional.softplus(module.scale_layer(x)) + module.eps)
    (loc.sum() + scale.sum()).backward()
    assert x.grad is not None and all(g is not None for g in _grads(module))
    assert_reference(f"gaussian_parameter_head_{transform}", {"loc": loc, "scale": scale})


def test_gaussian_parameter_head_errors() -> None:
    with pytest.raises(ValueError):
        GaussianParameterHead(2, 2, eps=0.0)
    with pytest.raises(ValueError):
        GaussianParameterHead(2, 2, scale_transform="relu")  # type: ignore[arg-type]
    big = GaussianParameterHead(1, 1)
    with torch.no_grad():
        big.scale_layer.weight.fill_(1000.0)
        big.scale_layer.bias.zero_()
    assert torch.isfinite(big(torch.ones(1, 1))[1]).all()  # softplus is stable


# ---------------------------------------------------------------- mixer_block
def test_mixer_block_contract_and_reference() -> None:
    torch.manual_seed(0)
    module = MixerBlock(6, 3, 8, 0.0)
    expected = {
        "time_norm.weight", "time_norm.bias", "feature_norm.weight", "feature_norm.bias",
        "time_projection.weight", "time_projection.bias",
        "feature_in.weight", "feature_in.bias", "feature_out.weight", "feature_out.bias",
    }
    assert set(module.state_dict()) == expected
    assert module.time_norm.weight.shape == (6, 3) and module.time_projection.weight.shape == (6, 6)
    x = _rand(2, 6, 3).requires_grad_(True)
    out = module(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    out.square().sum().backward()
    assert x.grad is not None and all(g is not None for g in _grads(module))
    with torch.no_grad():  # zeroed residual branches give the identity
        module.time_projection.weight.zero_(); module.time_projection.bias.zero_()
        module.feature_out.weight.zero_(); module.feature_out.bias.zero_()
    torch.testing.assert_close(module(x.detach()), x.detach())
    assert_reference("mixer_block", {"out": out, "x_grad": x.grad})


# ---------------------------------------------------------------- masking
def test_triangular_causal_mask() -> None:
    mask = TriangularCausalMask(2, 4).mask
    assert mask.dtype == torch.bool and mask.shape == (2, 1, 4, 4)
    assert torch.equal(mask[0, 0], torch.triu(torch.ones(4, 4, dtype=torch.bool), 1))
    assert not mask.requires_grad


def test_prob_mask() -> None:
    index = torch.tensor([[[0, 2], [3, 1]]])  # [B=1, H=2, n_selected=2]
    scores = torch.zeros(1, 2, 2, 5)
    mask = ProbMask(1, 2, 4, index, scores).mask
    assert mask.dtype == torch.bool and mask.shape == scores.shape
    for h in range(2):
        for s in range(2):
            row = int(index[0, h, s])
            assert torch.equal(mask[0, h, s], torch.arange(5) > row)


def test_local_mask_and_reference() -> None:
    length = 8
    mask = LocalMask(2, length, length).mask
    assert mask.dtype == torch.bool and mask.shape == (2, 1, length, length)
    local = math.ceil(math.log2(length))
    for i in range(length):
        for j in range(length):
            assert bool(mask[0, 0, i, j]) == (j > i or j < i - local)
    assert not mask[0, 0].diagonal().any()
    assert_reference(
        "masking",
        {
            "causal": TriangularCausalMask(1, 5).mask.to(torch.uint8),
            "local": mask[0, 0].to(torch.uint8),
        },
    )


# ---------------------------------------------------------------- periodic_query_bank
def test_periodic_query_bank_contract_and_reference() -> None:
    module = PeriodicQueryBank(5, 3)
    assert set(module.state_dict()) == {"table"} and module.table.shape == (5, 3)
    assert not module.table.any()
    assert module(torch.tensor([0, 2]), 4).shape == (2, 4, 3)
    with torch.no_grad():
        module.table.copy_(_rand(5, 3))
    phases = torch.tensor([0, 3, -1])
    out = module(phases, 7)  # length > period wraps
    assert out.shape == (3, 7, 3) and out.dtype == module.table.dtype
    for b, phase in enumerate(phases.tolist()):
        for t in range(7):
            assert torch.equal(out[b, t], module.table[(phase + t) % 5])
    out.square().sum().backward()
    assert module.table.grad is not None and module.table.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        module(torch.zeros(2, 1, dtype=torch.long), 3)
    with pytest.raises(ValueError):
        module(phases, 0)
    with pytest.raises(ValueError):
        PeriodicQueryBank(0, 3)
    with pytest.raises(ValueError):
        PeriodicQueryBank(3, 0)
    assert_reference("periodic_query_bank", {"out": out.detach(), "grad": module.table.grad})


# ---------------------------------------------------------------- positional_encoding
_KINDS = [None, "zero", "zeros", "normal", "gauss", "uniform", "sincos", "lin1d", "exp1d", "lin2d", "exp2d"]


@pytest.mark.parametrize("kind", _KINDS)
def test_positional_encoding_shapes(kind: str | None) -> None:
    torch.manual_seed(0)
    table = positional_encoding(kind, True, 6, 4)
    assert isinstance(table, torch.nn.Parameter) and table.dtype == torch.float32 and table.device.type == "cpu"
    width = 1 if kind in {"zero", "normal", "gauss", "uniform", "lin1d", "exp1d"} else 4
    assert table.shape == (6, width)
    assert table.requires_grad == (kind is not None)
    assert torch.isfinite(table).all()
    assert not positional_encoding(kind, False, 6, 4).requires_grad


def test_positional_encoding_errors_and_reference() -> None:
    with pytest.raises(ValueError):
        positional_encoding("bogus", True, 4, 4)
    with pytest.raises(ValueError):
        positional_encoding("sincos", True, 0, 4)
    with pytest.raises(ValueError):
        positional_encoding("sincos", True, 4, 0)
    torch.manual_seed(0)
    first = positional_encoding("normal", True, 5, 3)
    torch.manual_seed(0)
    assert torch.equal(first, positional_encoding("normal", True, 5, 3))
    sincos = positional_encoding("sincos", True, 6, 5)
    # standardized: zero mean, std = 0.1 for non-degenerate tables
    assert abs(float(sincos.mean())) < 1e-6 and abs(float(sincos.std()) - 0.1) < 1e-5
    assert_reference(
        "positional_encoding",
        {
            "sincos": sincos.detach(),
            "lin2d": positional_encoding("lin2d", True, 5, 3).detach(),
            "exp1d": positional_encoding("exp1d", True, 5, 3).detach(),
        },
    )

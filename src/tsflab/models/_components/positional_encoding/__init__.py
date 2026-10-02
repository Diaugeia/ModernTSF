"""Position-table construction for patch-based sequence encoders."""

from __future__ import annotations

import math

import torch
from torch import nn


def _standardize(table: torch.Tensor) -> torch.Tensor:
    deviation = table.std(unbiased=True)
    if float(deviation) == 0.0:
        return table - table.mean()
    return (table - table.mean()) / (10.0 * deviation)


def _sinusoidal(length: int, width: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    pairs = torch.arange(0, width, 2, dtype=torch.float32)
    angular = positions * torch.exp(-math.log(10_000.0) * pairs / width)[None, :]
    table = torch.empty(length, width)
    table[:, 0::2] = angular.sin()
    if width > 1:
        table[:, 1::2] = angular[:, : table[:, 1::2].shape[1]].cos()
    return _standardize(table)


def _coordinate(length: int, width: int, power: float) -> torch.Tensor:
    time = torch.linspace(0.0, 1.0, length)[:, None].pow(power)
    if width == 1:
        return _standardize(2.0 * time - 1.0)
    feature = torch.linspace(0.0, 1.0, width)[None, :].pow(power)
    return _standardize(2.0 * time * feature - 1.0)


def positional_encoding(
    kind: str | None,
    learnable: bool,
    length: int,
    width: int,
) -> nn.Parameter:
    """Create a positional table using the repository's stable public modes.

    Naming caveat: ``kind="zero"`` is NOT all zeros. It is a width-1 column of
    small uniform noise in ``[-0.02, 0.02]`` broadcast over features, and
    ``"zeros"`` is a full ``[length, width]`` table of the same noise. The
    legacy strings are kept so existing configs and checkpoints keep working;
    ``"small_uniform"`` is accepted as a clearer alias of ``"zero"`` and
    ``"small_uniform_2d"`` of ``"zeros"``.
    """
    if length < 1 or width < 1:
        raise ValueError("length and width must be positive")
    if kind is None:
        table = torch.empty(length, width).uniform_(-0.02, 0.02)
        learnable = False
    elif kind in {"zero", "small_uniform", "normal", "gauss", "uniform"}:
        table = torch.empty(length, 1)
        if kind in {"zero", "small_uniform"}:
            table.uniform_(-0.02, 0.02)
        elif kind in {"normal", "gauss"}:
            table.normal_(mean=0.0, std=0.1)
        else:
            table.uniform_(0.0, 0.1)
    elif kind in {"zeros", "small_uniform_2d"}:
        table = torch.empty(length, width).uniform_(-0.02, 0.02)
    elif kind == "sincos":
        table = _sinusoidal(length, width)
    elif kind in {"lin1d", "exp1d", "lin2d", "exp2d"}:
        table_width = width if kind.endswith("2d") else 1
        power = 0.5 if kind.startswith("exp") else 1.0
        table = _coordinate(length, table_width, power)
    else:
        raise ValueError(f"unsupported positional encoding: {kind!r}")
    return nn.Parameter(table, requires_grad=bool(learnable))

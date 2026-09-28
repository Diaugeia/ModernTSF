"""Kernel-free selective state-space blocks shared by Mamba forecasters.

The implementation follows the recurrence used by the repository's
MambaSimple port. It deliberately avoids ``mamba_ssm`` and CUDA-only kernels,
so every consumer receives the same portable tensor and gradient contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the final feature dimension."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MambaBlock(nn.Module):
    """Pure-PyTorch selective state-space mixer with a causal depthwise convolution."""

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        dt_rank: int,
        d_conv: int,
        d_state: int,
    ) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        state = repeat(torch.arange(1, d_state + 1), "n -> d n", d=d_inner).float()
        self.A_log = nn.Parameter(torch.log(state))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the selective state-space mixer to ``(batch, length, width)``."""
        length = x.shape[1]
        x, residual = self.in_proj(x).split([self.d_inner, self.d_inner], dim=-1)
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :length]
        x = F.silu(rearrange(x, "b d l -> b l d"))
        return self.out_proj(self.ssm(x) * F.silu(residual))

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Discretize and evaluate the input-dependent state-space recurrence."""
        state_size = self.A_log.shape[1]
        a = -torch.exp(self.A_log.float())
        d = self.D.float()
        delta, b, c = self.x_proj(x).split(
            [self.dt_rank, state_size, state_size], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))
        return self.selective_scan(x, delta, a, b, c, d)

    @staticmethod
    def selective_scan(
        u: torch.Tensor,
        delta: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the sequential selective scan without custom kernels."""
        batch, length, width = u.shape
        state_size = a.shape[1]
        delta_a = torch.exp(einsum(delta, a, "b l d, d n -> b l d n"))
        delta_b_u = einsum(delta, b, u, "b l d, b l n, b l d -> b l d n")
        state = torch.zeros((batch, width, state_size), device=delta_a.device)
        outputs = []
        for index in range(length):
            state = delta_a[:, index] * state + delta_b_u[:, index]
            outputs.append(einsum(state, c[:, index, :], "b d n, b n -> b d"))
        return torch.stack(outputs, dim=1) + u * d


class MambaResidualBlock(nn.Module):
    """Pre-normalized residual wrapper around :class:`MambaBlock`."""

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        dt_rank: int,
        d_conv: int,
        d_state: int,
    ) -> None:
        super().__init__()
        self.mixer = MambaBlock(d_model, d_inner, dt_rank, d_conv, d_state)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(self.norm(x)) + x

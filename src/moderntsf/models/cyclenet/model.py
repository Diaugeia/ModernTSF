"""Paper-driven local implementation of residual cycle forecasting."""

from __future__ import annotations

import torch
from torch import nn

from moderntsf.models._components.channel_wise_linear import ChannelWiseLinear
from moderntsf.models._components.revin import RevIN


class Model(nn.Module):
    """Remove a learnable recurrent cycle, forecast residuals, and restore it."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, cycle: int = 24,
                 model_type: str = "linear", d_model: int = 512,
                 use_revin: bool = True) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, cycle, d_model) < 1:
            raise ValueError("lengths, channels, cycle, and d_model must be positive")
        if model_type not in {"linear", "mlp"}:
            raise ValueError("model_type must be 'linear' or 'mlp'")
        self.seq_len, self.pred_len, self.enc_in, self.cycle = seq_len, pred_len, enc_in, cycle
        self.cycle_pattern = nn.Parameter(torch.zeros(cycle, enc_in))
        self.normalization = RevIN(enc_in, affine=False, enabled=use_revin)
        self.backbone = (
            ChannelWiseLinear(seq_len, pred_len, enc_in, individual=False)
            if model_type == "linear"
            else nn.Sequential(nn.Linear(seq_len, d_model), nn.ReLU(), nn.Linear(d_model, pred_len))
        )

    def _phase(self, marks: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
        if marks is None or marks.ndim != 3 or marks.shape[-1] < 6:
            return torch.zeros(batch, dtype=torch.long, device=device)
        stamp = marks[:, -1]
        weekday, hour = stamp[:, 3], stamp[:, 4]
        if self.cycle == 7:
            phase = weekday
        elif self.cycle == 168:
            phase = weekday * 24 + hour
        else:
            phase = hour
        return phase.long().remainder(self.cycle)

    def _cycle_values(self, phase: torch.Tensor, length: int) -> torch.Tensor:
        offsets = torch.arange(length, device=phase.device)
        indices = (phase[:, None] + offsets[None, :]).remainder(self.cycle)
        return self.cycle_pattern[indices]

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec, x_mark_dec
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("x_enc does not match configured time/channel dimensions")
        normalized = self.normalization(x_enc, "norm")
        end_phase = self._phase(x_mark_enc, x_enc.shape[0], x_enc.device)
        history_phase = (end_phase - self.seq_len + 1).remainder(self.cycle)
        history_cycle = self._cycle_values(history_phase, self.seq_len)
        future_cycle = self._cycle_values(end_phase + 1, self.pred_len)
        residual = normalized - history_cycle
        forecast = self.backbone(residual.transpose(1, 2)).transpose(1, 2)
        return self.normalization(forecast + future_cycle, "denorm")

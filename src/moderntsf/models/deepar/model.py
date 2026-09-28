"""Clean-room DeepAR implementation derived from the likelihood factorization."""

from __future__ import annotations

import torch
from torch import nn

from moderntsf.models._components.gaussian_parameter_head import GaussianParameterHead


class Model(nn.Module):
    """Global autoregressive LSTM with an independent Gaussian likelihood."""

    output_type = "distribution"
    distribution_family = "gaussian"

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        label_len: int = 0,
        features: str = "M",
        embedding_size: int = 32,
        hidden_size: int = 64,
        num_layers: int = 2,
        cov_feat_size: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        del label_len
        if min(
            seq_len, pred_len, enc_in, embedding_size, hidden_size, num_layers
        ) < 1:
            raise ValueError("lengths, channels, and recurrent widths must be positive")
        if cov_feat_size < 0:
            raise ValueError("cov_feat_size must be non-negative")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if features not in {"M", "S", "MS"}:
            raise ValueError("features must be M, S, or MS")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.features = features
        self.cov_feat_size = cov_feat_size
        self.value_embedding = nn.Linear(1, embedding_size)
        self.recurrent = nn.LSTM(
            embedding_size + cov_feat_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.likelihood = GaussianParameterHead(hidden_size, 1, eps=1e-6)

    def _covariates(
        self,
        marks: torch.Tensor | None,
        batch: int,
        steps: int,
        channels: int,
    ) -> torch.Tensor:
        if self.cov_feat_size == 0:
            return self.value_embedding.weight.new_empty(batch * channels, steps, 0)
        if marks is None:
            base = self.value_embedding.weight.new_zeros(
                batch, steps, self.cov_feat_size
            )
        else:
            if marks.ndim != 3 or marks.shape[:2] != (batch, steps):
                raise ValueError(f"marks must have shape [batch, {steps}, features]")
            base = marks[..., : self.cov_feat_size]
            if base.shape[-1] < self.cov_feat_size:
                base = torch.nn.functional.pad(
                    base, (0, self.cov_feat_size - base.shape[-1])
                )
        return base.unsqueeze(1).expand(-1, channels, -1, -1).reshape(
            batch * channels, steps, self.cov_feat_size
        )

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"x_enc must have shape [batch, {self.seq_len}, {self.enc_in}]"
            )
        batch = x_enc.shape[0]
        channels = self.enc_in
        history_cov = self._covariates(
            x_mark_enc, batch, self.seq_len, channels
        )
        history = x_enc.transpose(1, 2).reshape(
            batch * channels, self.seq_len, 1
        )
        encoded = torch.cat((self.value_embedding(history), history_cov), dim=-1)
        _, state = self.recurrent(encoded)

        if x_mark_dec is not None:
            future_marks = x_mark_dec[:, -self.pred_len :]
            if future_marks.shape[1] != self.pred_len:
                raise ValueError("x_mark_dec does not cover the forecast horizon")
        else:
            future_marks = None
        future_cov = self._covariates(
            future_marks, batch, self.pred_len, channels
        )
        feedback = history[:, -1:]
        locations: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        for step in range(self.pred_len):
            recurrent_input = torch.cat(
                (
                    self.value_embedding(feedback),
                    future_cov[:, step : step + 1],
                ),
                dim=-1,
            )
            recurrent_output, state = self.recurrent(recurrent_input, state)
            location, scale = self.likelihood(recurrent_output[:, 0])
            locations.append(location)
            scales.append(scale)
            # Deterministic inference uses the likelihood mean as the next input.
            feedback = location.unsqueeze(1)
        location = torch.stack(locations, dim=1).view(
            batch, channels, self.pred_len
        )
        scale = torch.stack(scales, dim=1).view(batch, channels, self.pred_len)
        location = location.transpose(1, 2)
        scale = scale.transpose(1, 2)
        if self.features == "MS":
            location = location[..., -1:]
            scale = scale[..., -1:]
        return torch.stack((location, scale), dim=-1)

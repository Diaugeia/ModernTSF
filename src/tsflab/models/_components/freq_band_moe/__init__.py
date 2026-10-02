"""Frequency-band decomposition mixture of experts.

Splits the ``rfft`` spectrum of a normalized ``(batch, channels, time)`` series
into ``expert_num`` contiguous frequency bands using learned, sorted band
boundaries, then combines the (masked) per-band spectra back into a single
signal with input-conditioned softmax gating weights. Each frequency band acts
as one "expert"; the gate decides, per input, how much each band contributes
to the reconstructed signal. This is a paper-neutral building block for
frequency-domain mixture-of-experts forecasters.
"""

from __future__ import annotations

import torch
from torch import nn


class FrequencyBandMixtureOfExperts(nn.Module):
    """Decompose a series into learned frequency bands and gate their mixture."""

    def __init__(self, expert_num: int, seq_len: int) -> None:
        super().__init__()
        if expert_num < 1:
            raise ValueError("expert_num must be positive")
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        self.expert_num = expert_num
        self.seq_len = seq_len
        self.freq_len = seq_len // 2 + 1
        self.band_boundaries = nn.Parameter(torch.rand(max(expert_num - 1, 0)))
        self.gating_network = nn.Sequential(
            nn.Linear(self.freq_len, self.freq_len),
            nn.ReLU(),
            nn.Linear(self.freq_len, expert_num),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize, band-decompose, and gate-combine ``x`` shaped ``[B, C, T]``.

        Returns ``(combined_output, boundaries, gating_scores)`` where
        ``combined_output`` is restored to the input's instance scale.
        """
        if x.ndim != 3 or x.shape[-1] != self.seq_len:
            raise ValueError(f"expected [batch, channels, {self.seq_len}], got {tuple(x.shape)}")
        x_mean = torch.mean(x, dim=2, keepdim=True)
        centered = x - x_mean
        x_var = torch.var(centered, dim=2, keepdim=True) + 1e-5
        normalized = centered / torch.sqrt(x_var)

        freq_x = torch.fft.rfft(normalized, dim=-1)
        total_freq_size = freq_x.size(-1)

        if self.expert_num > 1:
            boundaries = torch.sort(torch.sigmoid(self.band_boundaries)).values
        else:
            boundaries = self.band_boundaries
        boundaries = torch.cat(
            [
                torch.tensor([0.0], device=boundaries.device, dtype=boundaries.dtype),
                boundaries,
                torch.tensor([1.0], device=boundaries.device, dtype=boundaries.dtype),
            ]
        )
        indices = (boundaries * total_freq_size).long()
        indices[-1] = total_freq_size

        components = []
        for band in range(self.expert_num):
            start_idx = int(indices[band].item())
            end_idx = int(indices[band + 1].item())
            freq_mask = torch.zeros_like(freq_x)
            if end_idx > start_idx:
                freq_mask[:, :, start_idx:end_idx] = 1
            components.append((freq_x * freq_mask).unsqueeze(-1))
        components = torch.cat(components, dim=-1)

        gating_input = freq_x.abs().mean(dim=1)
        gating_scores = torch.softmax(self.gating_network(gating_input), dim=-1)

        combined_freq = torch.sum(components * gating_scores.unsqueeze(1).unsqueeze(2), dim=-1)
        combined_output = torch.fft.irfft(combined_freq, n=self.seq_len)
        combined_output = combined_output * torch.sqrt(x_var) + x_mean
        return combined_output, boundaries, gating_scores


__all__ = ["FrequencyBandMixtureOfExperts"]

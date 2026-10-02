"""Equivalence test for extracting TSMixer's block into ``_components.mixer_block``.

This freezes a copy of the pre-extraction ``MixerBlock``/``Model`` classes as
they existed in ``models/tsmixer/model.py`` before the shared
``tsflab.models._components.mixer_block.MixerBlock`` component was introduced, and
checks that the current, component-consuming implementation is behaviorally
identical: same ``state_dict()`` keys/shapes, same forward output, and same
gradients, from a fixed seed and fixed input.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from tsflab.models._components.channel_wise_linear import ChannelWiseLinear
from tsflab.models.tsmixer.model import Model as CurrentTSMixer


class _ReferenceMixerBlock(nn.Module):
    """Frozen copy of the original model-local ``MixerBlock`` (pre-extraction)."""

    def __init__(self, seq_len: int, channels: int, hidden: int, dropout: float) -> None:
        super().__init__()
        normalized_shape = (seq_len, channels)
        self.time_norm = nn.LayerNorm(normalized_shape)
        self.feature_norm = nn.LayerNorm(normalized_shape)
        self.time_projection = nn.Linear(seq_len, seq_len)
        self.feature_in = nn.Linear(channels, hidden)
        self.feature_out = nn.Linear(hidden, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_input = self.time_norm(x).transpose(1, 2)
        time_delta = self.dropout(self.activation(self.time_projection(time_input)))
        x = x + time_delta.transpose(1, 2)
        feature_input = self.feature_norm(x)
        feature_delta = self.feature_out(
            self.dropout(self.activation(self.feature_in(feature_input)))
        )
        return x + self.dropout(feature_delta)


class _ReferenceTSMixer(nn.Module):
    """Frozen copy of the original model-local ``Model`` (pre-extraction)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int,
        e_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.blocks = nn.ModuleList(
            _ReferenceMixerBlock(seq_len, enc_in, d_model, dropout) for _ in range(e_layers)
        )
        self.projection = ChannelWiseLinear(seq_len, pred_len, enc_in)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        hidden = x_enc
        for block in self.blocks:
            hidden = block(hidden)
        return self.projection(hidden.transpose(1, 2)).transpose(1, 2)


class MixerBlockExtractionEquivalenceTest(unittest.TestCase):
    """The extracted ``mixer_block`` component preserves TSMixer's behavior exactly."""

    def _build_pair(self):
        torch.manual_seed(0)
        reference = _ReferenceTSMixer(
            seq_len=16, pred_len=8, enc_in=3, d_model=6, e_layers=2, dropout=0.0
        )
        torch.manual_seed(0)
        current = CurrentTSMixer(
            seq_len=16, pred_len=8, enc_in=3, d_model=6, e_layers=2, dropout=0.0
        )
        return reference, current

    def test_state_dict_keys_and_shapes_match(self) -> None:
        reference, current = self._build_pair()
        reference_state = reference.state_dict()
        current_state = current.state_dict()
        self.assertEqual(set(reference_state.keys()), set(current_state.keys()))
        for key, value in reference_state.items():
            self.assertEqual(
                value.shape, current_state[key].shape, msg=f"shape mismatch at {key}"
            )

    def test_forward_outputs_match(self) -> None:
        reference, current = self._build_pair()
        reference.eval()
        current.eval()
        torch.manual_seed(1)
        x = torch.randn(4, 16, 3)
        with torch.no_grad():
            reference_out = reference(x)
            current_out = current(x)
        self.assertTrue(torch.allclose(reference_out, current_out, atol=1e-6))

    def test_gradients_match(self) -> None:
        reference, current = self._build_pair()
        torch.manual_seed(2)
        x = torch.randn(4, 16, 3)

        reference_out = reference(x)
        reference_out.sum().backward()
        current_out = current(x)
        current_out.sum().backward()

        reference_grads = {
            name: param.grad.clone()
            for name, param in reference.named_parameters()
            if param.grad is not None
        }
        current_grads = {
            name: param.grad.clone()
            for name, param in current.named_parameters()
            if param.grad is not None
        }
        self.assertEqual(set(reference_grads.keys()), set(current_grads.keys()))
        for name, grad in reference_grads.items():
            self.assertTrue(
                torch.allclose(grad, current_grads[name], atol=1e-6),
                msg=f"gradient mismatch at {name}",
            )


if __name__ == "__main__":
    unittest.main()

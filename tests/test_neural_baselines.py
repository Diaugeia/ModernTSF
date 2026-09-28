"""Equation, structure, and runtime tests for five neural baselines."""

from __future__ import annotations

import copy
import unittest

import torch
import torch.nn.functional as F

from moderntsf.models.gru_forecaster_ts.model import Model as GRUForecaster
from moderntsf.models.lstm_forecaster_ts.model import Model as LSTMForecaster
from moderntsf.models.mlp_forecaster_ts.model import Model as MLPForecaster
from moderntsf.models.rnn_forecaster_ts.model import Model as RNNForecaster
from moderntsf.models.tcn_forecaster_ts.model import CausalConv1d, Model as TCNForecaster


CASES = {
    "RNNForecasterTS": lambda length, horizon, channels: RNNForecaster(length, horizon, channels, d_model=5),
    "GRUForecasterTS": lambda length, horizon, channels: GRUForecaster(length, horizon, channels, d_model=5),
    "LSTMForecasterTS": lambda length, horizon, channels: LSTMForecaster(length, horizon, channels, d_model=5),
    "MLPForecasterTS": lambda length, horizon, channels: MLPForecaster(length, horizon, channels, d_model=7),
    "TCNForecasterTS": lambda length, horizon, channels: TCNForecaster(length, horizon, channels, d_model=5, num_layers=2),
}


def _manual_recurrence(module, x: torch.Tensor) -> torch.Tensor:
    """Evaluate PyTorch's documented one-layer RNN/GRU/LSTM equations."""
    hidden = x.new_zeros(x.size(0), module.hidden_size)
    cell = torch.zeros_like(hidden)
    for token in x.unbind(dim=1):
        input_terms = F.linear(token, module.weight_ih_l0, module.bias_ih_l0)
        hidden_terms = F.linear(hidden, module.weight_hh_l0, module.bias_hh_l0)
        if isinstance(module, torch.nn.RNN):
            hidden = torch.tanh(input_terms + hidden_terms)
        elif isinstance(module, torch.nn.GRU):
            input_reset, input_update, input_new = input_terms.chunk(3, dim=-1)
            hidden_reset, hidden_update, hidden_new = hidden_terms.chunk(3, dim=-1)
            reset = torch.sigmoid(input_reset + hidden_reset)
            update = torch.sigmoid(input_update + hidden_update)
            candidate = torch.tanh(input_new + reset * hidden_new)
            hidden = (1.0 - update) * candidate + update * hidden
        else:
            input_gate, forget_gate, candidate, output_gate = (input_terms + hidden_terms).chunk(4, dim=-1)
            cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
            hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
    return hidden


class NeuralBaselineEquationTests(unittest.TestCase):
    def test_recurrent_cells_follow_documented_equations(self) -> None:
        torch.manual_seed(19)
        x = torch.randn(2, 4, 3)
        for model_class in (RNNForecaster, GRUForecaster, LSTMForecaster):
            with self.subTest(cell=model_class.__name__):
                model = model_class(4, 2, 3, d_model=5, use_revin=False)
                _, state = model.encoder(x)
                actual = state[0][-1] if isinstance(state, tuple) else state[-1]
                torch.testing.assert_close(actual, _manual_recurrence(model.encoder, x))

    def test_mlp_is_channel_wise_lag_mapping(self) -> None:
        model = MLPForecaster(2, 1, 2, d_model=1, dropout=0.0, use_revin=False)
        with torch.no_grad():
            model.network[0].weight.copy_(torch.tensor([[2.0, -1.0]]))
            model.network[0].bias.copy_(torch.tensor([0.5]))
            model.network[-1].weight.copy_(torch.tensor([[3.0]]))
            model.network[-1].bias.copy_(torch.tensor([-2.0]))
        x = torch.tensor([[[1.0, 4.0], [3.0, 2.0]]])
        expected = (3.0 * F.gelu(torch.tensor([[-0.5, 6.5]])) - 2.0).reshape(1, 1, 2)
        torch.testing.assert_close(model(x), expected)

    def test_causal_convolution_has_no_future_leakage(self) -> None:
        convolution = CausalConv1d(1, 1, kernel_size=3, dilation=2)
        with torch.no_grad():
            convolution.weight.fill_(1.0)
            convolution.bias.zero_()
        first = torch.arange(7, dtype=torch.float32).reshape(1, 1, 7)
        second = first.clone()
        second[..., 5:] += 1000.0
        torch.testing.assert_close(convolution(first)[..., :5], convolution(second)[..., :5])


class NeuralBaselineContractTests(unittest.TestCase):
    def test_complete_runtime_contract(self) -> None:
        torch.manual_seed(733)
        for name, factory in CASES.items():
            with self.subTest(model=name):
                model = factory(4, 3, 2).cpu().eval()
                x = torch.randn(2, 4, 2, requires_grad=True)
                marks = torch.randn(2, 4, 3)
                adjacency = torch.eye(2)
                output = model(x, marks, adjacency)
                self.assertEqual(tuple(output.shape), (2, 3, 2))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())
                for parameter_name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, parameter_name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)
                    self.assertGreater(parameter.grad.abs().max().item(), 0.0, parameter_name)

                clone = factory(4, 3, 2).cpu().eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()))
                torch.testing.assert_close(clone(x.detach()), output.detach())
                self.assertEqual(tuple(model(torch.randn(1, 4, 2)).shape), (1, 3, 2))
                boundary = factory(1, 1, 2).cpu().eval()
                self.assertEqual(tuple(boundary(torch.randn(1, 1, 2)).shape), (1, 1, 2))
                with self.assertRaises(ValueError):
                    model(torch.randn(1, 3, 2))
                torch.testing.assert_close(
                    model(x.detach(), marks, adjacency), model(x.detach())
                )


if __name__ == "__main__":
    unittest.main()

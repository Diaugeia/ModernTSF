"""Paper-structure and reference checks for MambaTS."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from tsflab.models.mambats.model import (
    Model,
    TemporalMambaStack,
    anneal_scan_path,
    greedy_scan_path,
)
from tsflab.models.mambats.spec import SPEC, training_objective


def tiny(**overrides) -> Model:
    options = dict(
        seq_len=24, pred_len=6, enc_in=4, d_model=16, e_layers=2, patch_len=8,
        stride=8, dropout=0.0, atsp_steps=200,
    )
    options.update(overrides)
    torch.manual_seed(0)
    return Model(**options).eval()


def test_variable_major_scan_is_causal_over_variables() -> None:
    model = tiny(vpt_mode=0)
    x = torch.randn(2, 24, 4)
    base = model(x)
    later = x.clone()
    later[:, :, 2] += torch.randn(2, 24)
    changed = model(later)
    # tokens are laid out variable-major: earlier variables never see later ones
    torch.testing.assert_close(changed[:, :, :2], base[:, :, :2])
    assert not torch.allclose(changed[:, :, 2:], base[:, :, 2:])
    earlier = x.clone()
    earlier[:, :, 0] += torch.randn(2, 24)
    assert not torch.allclose(model(earlier)[:, :, 1:], base[:, :, 1:])


def test_inference_order_controls_which_variable_is_scanned_first() -> None:
    model = tiny()
    model._order = torch.tensor([2, 0, 3, 1])
    model._order_key = int(model.scan_updates)
    x = torch.randn(2, 24, 4)
    base = model(x)
    other = x.clone()
    other[:, :, 3] += torch.randn(2, 24)
    changed = model(other)
    # variable 2 is scanned first and 0 second, so neither sees variable 3
    torch.testing.assert_close(changed[:, :, [2, 0]], base[:, :, [2, 0]])
    assert not torch.allclose(changed[:, :, 1], base[:, :, 1])


def test_training_order_is_per_sample_permutation_and_output_is_restored() -> None:
    model = tiny(vpt_mode=1).train()
    x = torch.randn(3, 24, 4)
    out = model(x)
    assert out.shape == (3, 6, 4)
    transitions = model._transitions
    assert transitions.shape == (3, 4, 2)
    path = transitions[..., 1]
    for sample in path:
        assert sorted(sample.tolist()) == [0, 1, 2, 3]
    assert torch.equal(transitions[:, 0, 0], transitions[:, 0, 1])  # start self-loop
    assert torch.equal(transitions[:, 1:, 0], transitions[:, :-1, 1])


def test_unscrambled_output_independent_of_scan_order_without_mixing() -> None:
    # with a single layer-norm-free identity encoder the order cannot matter
    model = tiny(e_layers=1)
    with torch.no_grad():
        model.encoder.mixers[0].out_proj.weight.zero_()
    x = torch.randn(2, 24, 4)
    model._order, model._order_key = torch.tensor([3, 1, 0, 2]), 0
    first = model(x)
    model._order = torch.tensor([0, 1, 2, 3])
    torch.testing.assert_close(model(x), first)


def test_cost_update_credits_each_sample_to_its_own_transitions() -> None:
    model = tiny(enc_in=3).train()
    model._transitions = torch.tensor(
        [[[0, 0], [0, 1], [1, 2]], [[2, 2], [2, 0], [0, 1]]]
    )
    model.update_scan_costs(torch.tensor([1.0, 3.0]))
    scale = torch.tensor([1.0, 3.0]).std()
    total, count = model.scan_cost_sum, model.scan_cost_count
    assert count[0, 1] == 2 and count[1, 2] == 1 and count[2, 0] == 1
    assert count[0, 0] == 1 and count[2, 2] == 1 and count.sum() == 6
    torch.testing.assert_close(total[1, 2], torch.tensor(1.0) / scale)
    torch.testing.assert_close(total[2, 0], torch.tensor(3.0) / scale)
    torch.testing.assert_close(total[0, 1], torch.tensor(4.0) / scale)
    with pytest.raises(RuntimeError):
        model.update_scan_costs(torch.ones(2))


def test_cost_matrix_fills_unvisited_pairs_and_is_positive() -> None:
    model = tiny(enc_in=3)
    assert torch.all(model.scan_cost_matrix() > 0)
    model.scan_cost_sum[0, 1], model.scan_cost_count[0, 1] = 4.0, 2.0
    model.scan_cost_sum[1, 2], model.scan_cost_count[1, 2] = -1.0, 1.0
    matrix = model.scan_cost_matrix()
    assert matrix.min() == pytest.approx(1e-7)  # shifted so the cheapest pair is ~0
    assert matrix[0, 2] == pytest.approx(float(matrix[[0, 1], [1, 2]].mean()))


def _brute_force(cost: np.ndarray, start: int) -> float:
    rest = [i for i in range(len(cost)) if i != start]
    return min(
        cost[[start, *perm][:-1], [start, *perm][1:]].sum()
        for perm in itertools.permutations(rest)
    )


def test_scan_path_solvers_return_start_anchored_permutations() -> None:
    cost = np.random.default_rng(0).random((6, 6))
    np.fill_diagonal(cost, 0)
    greedy = greedy_scan_path(cost, 2)
    anneal = anneal_scan_path(cost, 2, steps=3000, seed=1)
    for path in (greedy, anneal):
        assert path[0] == 2 and sorted(path.tolist()) == list(range(6))
    cost_of = lambda p: cost[p[:-1], p[1:]].sum()
    assert cost_of(anneal) <= cost_of(greedy) + 1e-12
    assert cost_of(anneal) == pytest.approx(_brute_force(cost, 2))
    assert np.array_equal(anneal, anneal_scan_path(cost, 2, steps=3000, seed=1))


def test_scan_order_is_recomputed_after_cost_updates() -> None:
    model = tiny(enc_in=3, atsp_solver="GD")
    assert model.scan_order().tolist() == [0, 1, 2]
    model.scan_cost_sum[0, 0], model.scan_cost_count[0, 0] = 5.0, 1.0
    model.scan_cost_sum[1, 1], model.scan_cost_count[1, 1] = 1.0, 1.0  # cheapest start
    model.scan_cost_sum[1, 0], model.scan_cost_count[1, 0] = 0.1, 1.0
    model.scan_updates += 1
    order = model.scan_order().tolist()
    assert order[0] == 1 and order[1] == 0


def test_temporal_mamba_block_has_no_conv_and_dropout_on_selective_parameters() -> None:
    stack = TemporalMambaStack(16, 2, 4, 4, 2, dropout=0.3, use_conv=False)
    for mixer in stack.mixers:
        assert not mixer.use_conv and not hasattr(mixer, "conv1d")
        assert mixer.x_dropout.p == 0.3
    assert tiny(use_causal_conv=True).encoder.mixers[0].use_conv


def test_encoder_matches_add_then_norm_reference_recursion() -> None:
    """Reference: hidden/residual form where each block adds then normalises."""
    stack = TemporalMambaStack(16, 3, 4, 4, 2, dropout=0.0, use_conv=False).eval()
    tokens = torch.randn(2, 9, 16)
    hidden, residual = tokens, None
    for norm, mixer in zip(stack.norms, stack.mixers):
        residual = hidden if residual is None else hidden + residual
        hidden = mixer(norm(residual))
    expected = stack.final_norm(hidden + residual)
    torch.testing.assert_close(stack(tokens), expected)


def test_reference_initialisation_scales_residual_projection() -> None:
    deep = TemporalMambaStack(16, 16, 4, 4, 2, dropout=0.0, use_conv=False)
    shallow = TemporalMambaStack(16, 1, 4, 4, 2, dropout=0.0, use_conv=False)
    ratio = deep.mixers[0].out_proj.weight.abs().max() / shallow.mixers[0].out_proj.weight.abs().max()
    assert ratio < 0.6


def test_head_flattens_patch_major_and_matches_unfold_patching() -> None:
    model = tiny(patch_len=8, stride=8)
    assert model.num_patches == 3
    assert model.head.linear.in_features == 16 * 3
    assert tiny(seq_len=26, patch_len=8, stride=8).num_patches == 3  # remainder dropped


def test_training_objective_updates_costs_and_returns_criterion_loss() -> None:
    from tsflab.benchmark.runner.objective import TrainingBatch

    model = tiny().train()
    x, y = torch.randn(3, 24, 4), torch.randn(3, 24 + 6, 4)
    batch = TrainingBatch(
        x=x, x_mark=None, dec_inp=torch.zeros(3, 6, 4), y_mark=None,
        y=y[:, -6:], pred_len=6, features="M",
    )
    criterion = torch.nn.MSELoss()
    forecast, loss = training_objective(model, batch, criterion)
    torch.testing.assert_close(loss, criterion(forecast, batch.target))
    assert int(model.scan_updates) == 1 and model.scan_cost_count.sum() == 12
    loss.backward()
    fixed = tiny(vpt_mode=0).train()
    training_objective(fixed, batch, criterion)
    assert int(fixed.scan_updates) == 0


def test_spec_declares_components_and_objective() -> None:
    assert SPEC.components == ("flatten_forecast_head", "mamba", "revin")
    assert SPEC.training_objective is training_objective


def test_input_and_parameter_validation() -> None:
    with pytest.raises(ValueError):
        tiny()(torch.randn(2, 24, 3))
    with pytest.raises(ValueError):
        Model(seq_len=8, pred_len=2, enc_in=2, patch_len=16)
    with pytest.raises(ValueError):
        Model(seq_len=24, pred_len=2, enc_in=2, atsp_solver="LK")

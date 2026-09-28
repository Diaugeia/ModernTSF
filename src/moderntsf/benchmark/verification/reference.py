"""Reusable numerical reference-comparison harness for model verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import platform
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TensorComparison:
    """Numerical error summary for one output or gradient tensor."""

    passed: bool
    shape: tuple[int, ...]
    max_abs: float
    max_rel: float


@dataclass(frozen=True)
class ModeComparison:
    """Reference-comparison results collected in one model mode."""

    outputs: dict[str, TensorComparison]
    intermediates: dict[str, TensorComparison]
    input_gradients: dict[str, TensorComparison]
    parameter_gradients: dict[str, TensorComparison]

    @property
    def passed(self) -> bool:
        groups = (
            self.outputs,
            self.intermediates,
            self.input_gradients,
            self.parameter_gradients,
        )
        return all(item.passed for group in groups for item in group.values())


@dataclass(frozen=True)
class ReferenceComparisonReport:
    """Serializable result for one local/reference numerical comparison."""

    modes: dict[str, ModeComparison]
    seed: int
    atol: float
    rtol: float
    environment: dict[str, str]

    @property
    def passed(self) -> bool:
        return bool(self.modes) and all(mode.passed for mode in self.modes.values())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["passed"] = self.passed
        payload["modes"] = {
            name: {**asdict(result), "passed": result.passed}
            for name, result in self.modes.items()
        }
        return payload


def _tensor_tree(value: Any, prefix: str = "output") -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    if torch.is_tensor(value):
        tensors[prefix] = value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            tensors.update(_tensor_tree(item, f"{prefix}.{index}"))
    elif isinstance(value, dict):
        for key in sorted(value):
            tensors.update(_tensor_tree(value[key], f"{prefix}.{key}"))
    return tensors


def _compare(
    local: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> TensorComparison:
    if local.shape != reference.shape:
        return TensorComparison(False, tuple(local.shape), float("inf"), float("inf"))
    left = local.detach().cpu()
    right = reference.detach().cpu()
    # Casting a complex tensor to float silently discards its imaginary part.
    # Frequency-domain models such as FITS require both components to be part
    # of the comparison decision, so compare their real-valued views instead.
    if left.is_complex() or right.is_complex():
        left = (
            torch.view_as_real(left.resolve_conj())
            if left.is_complex()
            else torch.stack((left, torch.zeros_like(left)), dim=-1)
        )
        right = (
            torch.view_as_real(right.resolve_conj())
            if right.is_complex()
            else torch.stack((right, torch.zeros_like(right)), dim=-1)
        )
    left = left.float()
    right = right.float()
    absolute = (left - right).abs()
    denominator = torch.maximum(right.abs(), torch.full_like(right, atol))
    relative = absolute / denominator
    return TensorComparison(
        bool(torch.allclose(left, right, atol=atol, rtol=rtol, equal_nan=False)),
        tuple(left.shape),
        float(absolute.max()) if absolute.numel() else 0.0,
        float(relative.max()) if relative.numel() else 0.0,
    )


def _compare_trees(
    local: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> dict[str, TensorComparison]:
    if set(local) != set(reference):
        missing = sorted(set(local) ^ set(reference))
        raise ValueError(f"tensor structures differ at: {', '.join(missing)}")
    return {
        key: _compare(local[key], reference[key], atol=atol, rtol=rtol)
        for key in sorted(local)
    }


def copy_mapped_state(
    local: nn.Module,
    reference: nn.Module,
    state_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """Copy reference parameters and buffers into the local model.

    ``state_map`` maps local state-dict names to reference names. When omitted,
    both state dicts must have identical keys.
    """
    local_state = local.state_dict()
    reference_state = reference.state_dict()
    mapping = state_map or {name: name for name in local_state}
    if set(mapping) != set(local_state):
        missing = sorted(set(local_state) - set(mapping))
        extra = sorted(set(mapping) - set(local_state))
        raise ValueError(f"state map mismatch; missing={missing}, extra={extra}")
    copied: dict[str, torch.Tensor] = {}
    for local_name, reference_name in mapping.items():
        if reference_name not in reference_state:
            raise ValueError(f"reference state has no key {reference_name!r}")
        source = reference_state[reference_name]
        if local_state[local_name].shape != source.shape:
            raise ValueError(
                f"state shape mismatch for {local_name!r}: "
                f"{tuple(local_state[local_name].shape)} != {tuple(source.shape)}"
            )
        copied[local_name] = source.detach().clone()
    local.load_state_dict(copied, strict=True)
    return mapping


def _module(root: nn.Module, path: str) -> nn.Module:
    modules = dict(root.named_modules())
    if path not in modules:
        raise ValueError(f"model has no module path {path!r}")
    return modules[path]


def _run(
    model: nn.Module,
    inputs: tuple[Any, ...],
    module_paths: tuple[str, ...],
    *,
    seed: int,
    backward: bool,
) -> tuple[Any, dict[str, torch.Tensor], tuple[Any, ...]]:
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for path in module_paths:
        handles.append(
            _module(model, path).register_forward_hook(
                lambda _module, _args, output, name=path: captured.update(
                    _tensor_tree(output, name)
                )
            )
        )
    cloned = tuple(
        value.detach().clone().requires_grad_(backward and value.is_floating_point())
        if torch.is_tensor(value)
        else value
        for value in inputs
    )
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    try:
        output = model(*cloned)
        if backward:
            tensors = list(_tensor_tree(output).values())
            if not tensors:
                raise ValueError("model output contains no tensors")
            sum(tensor.float().sum() for tensor in tensors).backward()
    finally:
        for handle in handles:
            handle.remove()
    return output, captured, cloned


def compare_model_reference(
    local: nn.Module,
    reference: nn.Module,
    inputs: tuple[Any, ...],
    *,
    state_map: dict[str, str] | None = None,
    module_map: dict[str, str] | None = None,
    modes: tuple[str, ...] = ("eval", "train"),
    compare_gradients: bool = True,
    seed: int = 0,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> ReferenceComparisonReport:
    """Compare two implementations after copying an explicitly mapped state."""
    mapping = copy_mapped_state(local, reference, state_map)
    module_mapping = module_map or {}
    results: dict[str, ModeComparison] = {}
    for mode in modes:
        if mode not in {"eval", "train"}:
            raise ValueError(f"unsupported comparison mode {mode!r}")
        local.train(mode == "train")
        reference.train(mode == "train")
        local_output, local_mid, local_inputs = _run(
            local,
            inputs,
            tuple(module_mapping),
            seed=seed,
            backward=compare_gradients,
        )
        reference_output, reference_mid, reference_inputs = _run(
            reference,
            inputs,
            tuple(module_mapping.values()),
            seed=seed,
            backward=compare_gradients,
        )
        renamed_reference_mid = {}
        for local_name, reference_name in module_mapping.items():
            for key, tensor in reference_mid.items():
                if key == reference_name or key.startswith(f"{reference_name}."):
                    renamed_reference_mid[f"{local_name}{key[len(reference_name):]}"] = tensor
        input_gradients = {}
        if compare_gradients:
            local_grads = {
                f"input.{index}": value.grad
                for index, value in enumerate(local_inputs)
                if torch.is_tensor(value)
                and value.requires_grad
                and value.grad is not None
            }
            reference_grads = {
                f"input.{index}": value.grad
                for index, value in enumerate(reference_inputs)
                if torch.is_tensor(value)
                and value.requires_grad
                and value.grad is not None
            }
            input_gradients = _compare_trees(
                local_grads, reference_grads, atol=atol, rtol=rtol
            )
        local_parameters = dict(local.named_parameters())
        reference_parameters = dict(reference.named_parameters())
        parameter_gradients = {}
        if compare_gradients:
            local_grads = {}
            reference_grads = {}
            for local_name, reference_name in mapping.items():
                if local_name not in local_parameters:
                    continue
                local_grad = local_parameters[local_name].grad
                reference_grad = reference_parameters[reference_name].grad
                if local_grad is None or reference_grad is None:
                    continue
                local_grads[local_name] = local_grad
                reference_grads[local_name] = reference_grad
            parameter_gradients = _compare_trees(
                local_grads, reference_grads, atol=atol, rtol=rtol
            )
        results[mode] = ModeComparison(
            outputs=_compare_trees(
                _tensor_tree(local_output),
                _tensor_tree(reference_output),
                atol=atol,
                rtol=rtol,
            ),
            intermediates=_compare_trees(
                local_mid, renamed_reference_mid, atol=atol, rtol=rtol
            ),
            input_gradients=input_gradients,
            parameter_gradients=parameter_gradients,
        )
    return ReferenceComparisonReport(
        modes=results,
        seed=seed,
        atol=atol,
        rtol=rtol,
        environment={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(next(local.parameters(), torch.empty(0)).device),
        },
    )

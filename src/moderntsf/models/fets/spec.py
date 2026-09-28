"""Model specification for FeTS."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.fets.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 32
    patch_len: int = 16
    stride: int = 8
    fourier_order: int = 2
    polynomial_order: int = 2
    kernel_size: int = 3
    dropout: float = 0.0
    use_revin: bool = True


def build_model(cfg, params):
    """Construct FeTS from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params["enc_in"],
        d_model=params.get("d_model", 32), patch_len=params.get("patch_len", 16),
        stride=params.get("stride", 8), fourier_order=params.get("fourier_order", 2),
        polynomial_order=params.get("polynomial_order", 2), kernel_size=params.get("kernel_size", 3),
        dropout=params.get("dropout", 0.0), use_revin=bool(params.get("use_revin", True)),
    )


SPEC = ModelSpec(
    name='FeTS',
    module='moderntsf.models.fets',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/FeTS.toml',
    model_card='src/moderntsf/models/fets/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
        components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)

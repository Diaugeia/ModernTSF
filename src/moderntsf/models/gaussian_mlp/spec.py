"""Model specification for GaussianMLP."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.gaussian_mlp.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    hidden_size: int = Field(default=256, gt=0)
    num_layers: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    eps: float = Field(default=1e-6, gt=0.0)


def build_model(cfg, params):
    """Construct GaussianMLP from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], features=cfg.task.features, hidden_size=params.get('hidden_size', 256), num_layers=params.get('num_layers', 2), dropout=params.get('dropout', 0.1), eps=params.get('eps', 1e-06))
    )


SPEC = ModelSpec(
    name='GaussianMLP',
    module='moderntsf.models.gaussian_mlp',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/GaussianMLP.toml',
    model_card='src/moderntsf/models/gaussian_mlp/README.md',
    smoke_config='configs/runs/smoke_gaussian_mlp.toml',
    capabilities=frozenset(['distribution-output', 'time-series']),
    components=('gaussian_parameter_head',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)

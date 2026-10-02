"""Model specification for PAttn."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.pattn.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    d_model: int = Field(default=128, gt=0)
    n_heads: int = Field(default=8, gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


def build_model(cfg, params):
    """Construct PAttn from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        features=cfg.task.features,
        d_model=params.get("d_model", 128),
        n_heads=params.get("n_heads", 8),
        patch_len=params.get("patch_len", 16),
        stride=params.get("stride", 8),
        dropout=params.get("dropout", 0.1),
    )


SPEC = ModelSpec(
    name='PAttn',
    module='tsflab.models.pattn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/PAttn.toml',
    model_card='src/tsflab/models/pattn/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)

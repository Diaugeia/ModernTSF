"""Model specification for CrossLinear."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.crosslinear.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    patch_len: int = Field(default=16, gt=0)
    d_model: int = Field(default=32, gt=0)
    d_ff: int = Field(default=128, gt=0)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    beta: float = Field(default=0.5, ge=0.0, le=1.0)


def build_model(cfg, params):
    """Construct CrossLinear from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        patch_len=params.get("patch_len", 16),
        d_model=params.get("d_model", 32),
        d_ff=params.get("d_ff", 128),
        alpha=params.get("alpha", 0.5),
        beta=params.get("beta", 0.5),
    )


SPEC = ModelSpec(
    name='CrossLinear',
    module='tsflab.models.crosslinear',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/CrossLinear.toml',
    model_card='src/tsflab/models/crosslinear/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)

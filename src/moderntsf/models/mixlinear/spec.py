"""Model specification for MixLinear."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.mixlinear.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    downsample: int = Field(default=4, gt=0)
    segments: int = Field(default=4, gt=0)
    hidden_rank: int = Field(default=2, gt=0)
    spectral_rank: int = Field(default=2, gt=0)


def build_model(cfg, params):
    """Construct MixLinear from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        downsample=params.get("downsample", 4),
        segments=params.get("segments", 4),
        hidden_rank=params.get("hidden_rank", 2),
        spectral_rank=params.get("spectral_rank", 2),
    )


SPEC = ModelSpec(
    name='MixLinear',
    module='moderntsf.models.mixlinear',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/MixLinear.toml',
    model_card='src/moderntsf/models/mixlinear/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)

"""Model specification for PWS."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.pws.model import Model

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    period: int = Field(default=24, gt=0)
    patch_size: int = Field(default=6, gt=0)
    revin: bool = True
    affine: bool = False
    subtract_last: bool = False
    analysis_act: Literal["relu", "gelu", "silu", "tanh", "leaky_relu"] = "relu"
    analysis_hidden: list[int] = Field(default_factory=lambda: [512, 256])

    @field_validator("analysis_hidden")
    @classmethod
    def _positive_hidden_sizes(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values):
            raise ValueError("analysis_hidden entries must be positive")
        return values


def build_model(cfg, params):
    """Construct PWS from a validated run configuration."""
    return (
    Model(c_in=params['enc_in'], period=params.get('period', 24), seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, patch_size=params.get('patch_size', 6), revin=bool(params.get('revin', True)), affine=bool(params.get('affine', False)), subtract_last=bool(params.get('subtract_last', False)), analysis_act=params.get('analysis_act', 'relu'), analysis_hidden=params.get('analysis_hidden', [512, 256]))
    )


SPEC = ModelSpec(
    name='PWS',
    module='tsflab.models.pws',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/PWS.toml',
    model_card='src/tsflab/models/pws/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)

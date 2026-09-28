"""Model specification for TimeBase."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.timebase.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    period_len: int = Field(default=24, gt=0)
    basis_num: int = Field(default=6, gt=0)
    individual: bool = False
    orthogonal_weight: float = Field(default=0.08, ge=0.0)
    use_period_norm: bool = True


def build_model(cfg, params):
    """Construct TimeBase from a validated run configuration."""
    return (
        Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], period_len=params.get('period_len', 24), basis_num=params.get('basis_num', 6), individual=bool(params.get('individual', False)), orthogonal_weight=float(params.get('orthogonal_weight', 0.08)), use_period_norm=bool(params.get('use_period_norm', True)))
    )


SPEC = ModelSpec(
    name='TimeBase',
    module='moderntsf.models.timebase',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/TimeBase.toml',
    model_card='src/moderntsf/models/timebase/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)

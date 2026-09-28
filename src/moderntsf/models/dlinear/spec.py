"""Model specification for DLinear."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.dlinear.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    kernel_size: int = 25
    individual: bool = False


def build_model(cfg, params):
    """Construct DLinear from a validated run configuration."""
    return (
    Model(c_in=params['enc_in'], seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, kernel_size=params.get('kernel_size', 25), individual=params.get('individual', False))
    )


SPEC = ModelSpec(
    name='DLinear',
    module='moderntsf.models.dlinear',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/DLinear.toml',
    model_card='src/moderntsf/models/dlinear/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('dlinear',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)

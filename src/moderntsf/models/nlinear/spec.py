"""Model specification for NLinear."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.nlinear.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    individual: bool = False


def build_model(cfg, params):
    """Construct NLinear from a validated run configuration."""
    return (
    Model(c_in=params['enc_in'], seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, individual=bool(params.get('individual', False)))
    )


SPEC = ModelSpec(
    name='NLinear',
    module='moderntsf.models.nlinear',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/NLinear.toml',
    model_card='src/moderntsf/models/nlinear/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('channel_wise_linear', 'last_value_center'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)

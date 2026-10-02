"""Model specification for SparseTSF."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.sparsetsf.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    period: int = 24
    d_model: int = 64
    model_type: str = "linear"


def build_model(cfg, params):
    """Construct SparseTSF from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], period=params.get('period', 24), d_model=params.get('d_model', 64), model_type=params.get('model_type', 'linear'))
    )


SPEC = ModelSpec(
    name='SparseTSF',
    module='tsflab.models.sparsetsf',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/SparseTSF.toml',
    model_card='src/tsflab/models/sparsetsf/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)

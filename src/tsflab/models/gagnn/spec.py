"""Model specification for GAGNN."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.gagnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated GAGNN parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    d_model: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    group_num: int = 4


def build_model(cfg, params):
    """Construct GAGNN from a validated run configuration."""
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], adj_mx=params.get('adj_mx'), cov_dim=params.get('cov_dim', 2), d_model=params.get('d_model', 64), num_layers=params.get('num_layers', 3), dropout=params.get('dropout', 0.1), group_num=params.get('group_num', 4))


SPEC = ModelSpec(
    name='GAGNN',
    module='tsflab.models.gagnn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/GAGNN.toml',
    model_card='src/tsflab/models/gagnn/README.md',
    smoke_config=None,
    capabilities=frozenset(['covariate']),
    components=('marks',),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)

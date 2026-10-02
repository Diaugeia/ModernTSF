"""Model specification for PM25_GNN."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.pm25gnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated PM25_GNN parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    hid_dim: int = 64


def build_model(cfg, params):
    """Construct PM25_GNN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], adj_mx=params.get('adj_mx'), cov_dim=params.get('cov_dim', 2), hid_dim=params.get('hid_dim', 64))
    )


SPEC = ModelSpec(
    name='PM25_GNN',
    module='tsflab.models.pm25gnn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/PM25_GNN.toml',
    model_card='src/tsflab/models/pm25gnn/README.md',
    smoke_config=None,
    capabilities=frozenset(['covariate']),
    components=('marks',),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)

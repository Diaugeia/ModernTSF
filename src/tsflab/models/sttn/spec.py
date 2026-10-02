"""Model specification for STTN."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.sttn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STTN parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    d_model: int = 64
    mlp_expand: int = 4
    num_layers: int = 3
    dropout: float = 0.1


def build_model(cfg, params):
    """Construct STTN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"], adj_mx=params.get("adj_mx"),
        cov_dim=params.get("cov_dim", 2), d_model=params.get("d_model", 64),
        mlp_expand=params.get("mlp_expand", 4), num_layers=params.get("num_layers", 3),
        dropout=params.get("dropout", 0.1),
    )


SPEC = ModelSpec(
    name='STTN',
    module='tsflab.models.sttn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/STTN.toml',
    model_card='src/tsflab/models/sttn/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)

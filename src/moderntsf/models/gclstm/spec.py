"""Model specification for GCLSTM."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.gclstm.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    """Validated GCLSTM parameters supplied via ``model.params``."""

    enc_in: int = Field(ge=1)
    cov_dim: int = Field(default=2, ge=0)
    Ks: int = Field(default=2, ge=1)
    hidden_dim: int = Field(default=64, ge=1)


def build_model(cfg, params):
    """Construct GCLSTM from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], adj_mx=params.get('adj_mx'), cov_dim=params.get('cov_dim', 2), Ks=params.get('Ks', 2), hidden_dim=params.get('hidden_dim', 64))
    )


SPEC = ModelSpec(
    name='GCLSTM',
    module='moderntsf.models.gclstm',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/GCLSTM.toml',
    model_card='src/moderntsf/models/gclstm/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal', 'covariate']),
    components=('channel_alignment', 'graph_spectral', 'marks'),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)

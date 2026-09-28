"""Model specification for MGSFformer."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.mgsfformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated MGSFformer parameters supplied via ``model.params``."""

    enc_in: int
    IE_dim: int = 32
    dropout: float = 0.3
    num_head: int = 2


def build_model(cfg, params):
    """Construct MGSFformer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"], IE_dim=params.get("IE_dim", 32),
        dropout=params.get("dropout", 0.3), num_head=params.get("num_head", 2),
    )


SPEC = ModelSpec(
    name='MGSFformer',
    module='moderntsf.models.mgsfformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/MGSFformer.toml',
    model_card='src/moderntsf/models/mgsfformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('revin',),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)

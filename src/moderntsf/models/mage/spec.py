"""Model specification for MAGE."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.mage.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated MAGE parameters supplied via ``model.params``."""

    enc_in: int
    model_dim: int = 64
    recur_num: int = 8
    topk: int = 2
    node_dim: int = 16


def build_model(cfg, params):
    """Construct MAGE from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"], model_dim=params.get("model_dim", 64),
        recur_num=params.get("recur_num", 8), topk=params.get("topk", 2),
        node_dim=params.get("node_dim", 16),
    )


SPEC = ModelSpec(
    name='MAGE',
    module='moderntsf.models.mage',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/MAGE.toml',
    model_card='src/moderntsf/models/mage/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)

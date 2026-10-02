"""Model specification for PHAT."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.phat.model import Model

from typing import Optional

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    """Validated PHAT parameters supplied via ``model.params``."""

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    n_heads: int = Field(default=8, gt=0)
    d_layers: int = Field(default=1, gt=0)
    attn_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    ffn_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    ffn_expand_ratio: float = Field(default=2.66667, gt=0.0)
    period_topk: int = Field(default=1, gt=0)
    period_list: Optional[list[int]] = None


def build_model(cfg, params):
    """Construct PHAT from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 64),
        n_heads=params.get("n_heads", 8),
        d_layers=params.get("d_layers", 1),
        attn_dropout=params.get("attn_dropout", 0.1),
        ffn_dropout=params.get("ffn_dropout", 0.1),
        ffn_expand_ratio=params.get("ffn_expand_ratio", 2.66667),
        period_topk=params.get("period_topk", 1),
        period_list=params.get("period_list"),
    )


SPEC = ModelSpec(
    name='PHAT',
    module='tsflab.models.phat',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/PHAT.toml',
    model_card='src/tsflab/models/phat/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)

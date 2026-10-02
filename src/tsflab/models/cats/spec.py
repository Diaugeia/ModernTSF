"""Model specification for CATS."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.cats.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    patch_len: int = 24
    d_model: int = 128
    n_heads: int = 16
    d_ff: int = 256
    n_layers: int = 3
    dropout: float = 0.1
    stride: int = 24
    attn_dropout: float = 0.0
    query_independence: bool = False
    padding_patch: str | None = None
    store_attn: bool = False
    QAM_start: float = 0.1
    QAM_end: float = 0.5


def build_model(cfg, params):
    """Construct CATS from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], patch_len=params.get('patch_len', 24), d_model=params.get('d_model', 128), n_heads=params.get('n_heads', 16), d_ff=params.get('d_ff', 256), n_layers=params.get('n_layers', 3), dropout=params.get('dropout', 0.1), stride=params.get('stride', 24), attn_dropout=params.get('attn_dropout', 0.0), query_independence=params.get('query_independence', False), padding_patch=params.get('padding_patch'), store_attn=params.get('store_attn', False), QAM_start=params.get('QAM_start', 0.1), QAM_end=params.get('QAM_end', 0.5))
    )


SPEC = ModelSpec(
    name='CATS',
    module='tsflab.models.cats',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/CATS.toml',
    model_card='src/tsflab/models/cats/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)

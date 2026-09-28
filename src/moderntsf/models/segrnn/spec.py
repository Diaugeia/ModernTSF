"""Model specification for SegRNN."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.segrnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    seg_len: int = 24


def build_model(cfg, params):
    """Construct SegRNN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], d_model=params.get('d_model', 64), dropout=params.get('dropout', 0.1), seg_len=params.get('seg_len', 24))
    )


SPEC = ModelSpec(
    name='SegRNN',
    module='moderntsf.models.segrnn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/SegRNN.toml',
    model_card='src/moderntsf/models/segrnn/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('last_value_center',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)

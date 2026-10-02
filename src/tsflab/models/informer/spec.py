"""Model specification for Informer."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.informer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    dec_in: int | None = None
    c_out: int | None = None
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 256
    dropout: float = 0.1
    factor: int = 3
    activation: str = "gelu"
    distil: bool = True
    embed: str = "timeF"
    freq: str = "h"


def build_model(cfg, params):
    """Construct Informer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], dec_in=params.get('dec_in'), c_out=params.get('c_out'), d_model=params.get('d_model', 128), n_heads=params.get('n_heads', 8), e_layers=params.get('e_layers', 2), d_layers=params.get('d_layers', 1), d_ff=params.get('d_ff', 256), dropout=params.get('dropout', 0.1), factor=params.get('factor', 3), activation=params.get('activation', 'gelu'), distil=bool(params.get('distil', True)), embed=params.get('embed', 'timeF'), freq=params.get('freq', 'h'))
    )


SPEC = ModelSpec(
    name='Informer',
    module='tsflab.models.informer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/Informer.toml',
    model_card='src/tsflab/models/informer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed', 'self_attention_family', 'transformer_encdec'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)

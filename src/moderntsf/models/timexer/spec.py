"""Model specification for TimeXer."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.timexer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 256
    patch_len: int = 16
    dropout: float = 0.1
    activation: str = "gelu"
    use_norm: bool = True


def build_model(cfg, params):
    """Construct TimeXer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 128), n_heads=params.get('n_heads', 8), e_layers=params.get('e_layers', 2), d_ff=params.get('d_ff', 256), patch_len=params.get('patch_len', 16), dropout=params.get('dropout', 0.1), activation=params.get('activation', 'gelu'), use_norm=bool(params.get('use_norm', True)))
    )


SPEC = ModelSpec(
    name='TimeXer',
    module='moderntsf.models.timexer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/TimeXer.toml',
    model_card='src/moderntsf/models/timexer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)

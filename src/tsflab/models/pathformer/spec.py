"""Model specification for Pathformer."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.pathformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    layer_nums: int = 2
    k: int = 2
    num_experts: int = 4
    # Flat list of length layer_nums * num_experts, reshaped to per-layer
    # patch sizes. Each value must divide seq_len evenly.
    patch_size_list: list[int] = [16, 12, 8, 6, 16, 12, 8, 6]
    d_model: int = 16
    d_ff: int = 64
    residual_connection: int = 1
    revin: bool = True
    n_heads: int = 4
    dropout: float = 0.1


def build_model(cfg, params):
    """Construct Pathformer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, features=cfg.task.features, enc_in=params['enc_in'], layer_nums=params.get('layer_nums', 2), k=params.get('k', 2), num_experts=params.get('num_experts', 4), patch_size_list=params.get('patch_size_list', [16, 12, 8, 6, 16, 12, 8, 6]), d_model=params.get('d_model', 16), d_ff=params.get('d_ff', 64), residual_connection=params.get('residual_connection', 1), revin=bool(params.get('revin', True)), n_heads=params.get('n_heads', 4), dropout=params.get('dropout', 0.1))
    )


SPEC = ModelSpec(
    name='Pathformer',
    module='tsflab.models.pathformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/Pathformer.toml',
    model_card='src/tsflab/models/pathformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)

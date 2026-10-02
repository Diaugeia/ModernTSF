"""Model specification for QuantilePatchTST."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.quantile_patchtst.model import Model

from typing import Optional

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    patch_len: int = 16
    stride: int = 8
    padding_patch: str = "end"
    e_layers: int = 3
    d_model: int = 512
    n_heads: int = 8
    d_k: Optional[int] = None
    d_v: Optional[int] = None
    d_ff: int = 2048
    activation: str = "gelu"
    norm: str = "BatchNorm"
    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0
    res_dropout: float = 0.0
    proj_dropout: float = 0.0
    head_dropout: float = 0.0
    pre_norm: bool = False
    pe: str = "zeros"
    learn_pe: bool = False
    head_type: str = "flatten"
    individual: bool = False
    revin: bool = True
    affine: bool = False
    subtract_last: bool = False
    quantile_levels: list[float] | None = None


def build_model(cfg, params):
    """Construct QuantilePatchTST from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], features=cfg.task.features, patch_len=params.get('patch_len', 16), stride=params.get('stride', 8), padding_patch=params.get('padding_patch', 'end'), e_layers=params.get('e_layers', 3), d_model=params.get('d_model', 512), n_heads=params.get('n_heads', 8), d_k=params.get('d_k'), d_v=params.get('d_v'), d_ff=params.get('d_ff', 2048), activation=params.get('activation', 'gelu'), norm=params.get('norm', 'BatchNorm'), attn_dropout=params.get('attn_dropout', 0.0), res_dropout=params.get('res_dropout', 0.0), ffn_dropout=params.get('ffn_dropout', 0.0), proj_dropout=params.get('proj_dropout', 0.0), head_dropout=params.get('head_dropout', 0.0), pre_norm=bool(params.get('pre_norm', False)), pe=params.get('pe', 'zeros'), learn_pe=bool(params.get('learn_pe', False)), head_type=params.get('head_type', 'flatten'), individual=bool(params.get('individual', False)), revin=bool(params.get('revin', True)), affine=bool(params.get('affine', False)), subtract_last=bool(params.get('subtract_last', False)), quantile_levels=params.get('quantile_levels') or list(cfg.evaluation.quantile_levels))
    )


SPEC = ModelSpec(
    name='QuantilePatchTST',
    module='tsflab.models.quantile_patchtst',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/QuantilePatchTST.toml',
    model_card='src/tsflab/models/quantile_patchtst/README.md',
    smoke_config=None,
    capabilities=frozenset(['quantile-output', 'time-series']),
    components=('patchtst', 'quantile_head'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)

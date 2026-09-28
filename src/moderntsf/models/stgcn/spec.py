"""Model specification for STGCN."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.stgcn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STGCN parameters supplied via ``model.params``.

    ``enc_in`` (the number of spatial nodes ``N``) is required. ``num_nodes``
    and ``adj_mx`` are injected by the runner from the dataset and are NOT
    declared in the TOML.
    """

    enc_in: int
    input_dim: int = 3
    Kt: int = 3
    Ks: int = 3
    hidden_dim: int = 64
    bottleneck_dim: int = 16
    out_hidden_dim: int = 128
    act_func: str = "glu"
    graph_conv_type: str = "cheb_graph_conv"
    bias: bool = True
    droprate: float = 0.5


def build_model(cfg, params):
    """Construct STGCN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3), Kt=params.get('Kt', 3), Ks=params.get('Ks', 3), hidden_dim=params.get('hidden_dim', 64), bottleneck_dim=params.get('bottleneck_dim', 16), out_hidden_dim=params.get('out_hidden_dim', 128), act_func=params.get('act_func', 'glu'), graph_conv_type=params.get('graph_conv_type', 'cheb_graph_conv'), bias=params.get('bias', True), droprate=params.get('droprate', 0.5))
    )


SPEC = ModelSpec(
    name='STGCN',
    module='moderntsf.models.stgcn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/STGCN.toml',
    model_card='src/moderntsf/models/stgcn/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('graph_spectral', 'marks'),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)

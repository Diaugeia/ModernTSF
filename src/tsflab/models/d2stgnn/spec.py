"""Model specification for D2STGNN."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.d2stgnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated D2STGNN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``run_one``) and are therefore not declared here.
    """

    enc_in: int
    input_dim: int = 3
    num_feat: int = 1
    num_hidden: int = 16
    node_hidden: int = 8
    time_emb_dim: int = 8
    k_s: int = 2
    k_t: int = 3
    gap: int = 1
    num_layers: int = 2
    dropout: float = 0.1
    time_in_day_size: int = 288
    day_in_week_size: int = 7
    forecast_dim: int = 64
    output_hidden: int = 128


def build_model(cfg, params):
    """Construct D2STGNN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3), num_feat=params.get('num_feat', 1), num_hidden=params.get('num_hidden', 16), node_hidden=params.get('node_hidden', 8), time_emb_dim=params.get('time_emb_dim', 8), k_s=params.get('k_s', 2), k_t=params.get('k_t', 3), gap=params.get('gap', 1), num_layers=params.get('num_layers', 2), dropout=params.get('dropout', 0.1), time_in_day_size=params.get('time_in_day_size', 288), day_in_week_size=params.get('day_in_week_size', 7), forecast_dim=params.get('forecast_dim', 64), output_hidden=params.get('output_hidden', 128))
    )


SPEC = ModelSpec(
    name='D2STGNN',
    module='tsflab.models.d2stgnn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/D2STGNN.toml',
    model_card='src/tsflab/models/d2stgnn/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('adaptive_node_embedding_adjacency', 'graph_utils', 'marks'),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)

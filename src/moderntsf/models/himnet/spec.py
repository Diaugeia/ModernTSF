"""Model specification for HimNet."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.himnet.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated HimNet parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset and
    need not be declared in TOML. ``enc_in`` (= number of nodes ``N``) is the
    required channel count; the remaining fields carry modest defaults for a
    fast smoke run.
    """

    enc_in: int
    input_dim: int = 3
    output_dim: int = 1
    hidden_dim: int = 32
    num_layers: int = 1
    cheb_k: int = 2
    node_embedding_dim: int = 8
    st_embedding_dim: int = 8
    tod_embedding_dim: int = 8
    dow_embedding_dim: int = 8
    steps_per_day: int = 288
    use_teacher_forcing: bool = True


def build_model(cfg, params):
    """Construct HimNet from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3), output_dim=params.get('output_dim', 1), hidden_dim=params.get('hidden_dim', 32), num_layers=params.get('num_layers', 1), cheb_k=params.get('cheb_k', 2), node_embedding_dim=params.get('node_embedding_dim', 8), st_embedding_dim=params.get('st_embedding_dim', 8), tod_embedding_dim=params.get('tod_embedding_dim', 8), dow_embedding_dim=params.get('dow_embedding_dim', 8), steps_per_day=params.get('steps_per_day', 288), use_teacher_forcing=params.get('use_teacher_forcing', True))
    )


SPEC = ModelSpec(
    name='HimNet',
    module='moderntsf.models.himnet',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/HimNet.toml',
    model_card='src/moderntsf/models/himnet/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('adaptive_node_embedding_adjacency', 'marks'),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)

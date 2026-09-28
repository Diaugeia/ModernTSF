"""Model specification for AGCRN."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.agcrn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated AGCRN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset and
    are therefore not declared here. ``enc_in`` (= number of nodes ``N``) is the
    only required field; the remaining defaults are kept modest for a fast
    spatiotemporal smoke.
    """

    enc_in: int
    input_dim: int = 3
    rnn_units: int = 32
    embed_dim: int = 8
    num_layers: int = 1
    cheb_k: int = 2
    output_dim: int = 1


def build_model(cfg, params):
    """Construct AGCRN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3), rnn_units=params.get('rnn_units', 32), embed_dim=params.get('embed_dim', 8), num_layers=params.get('num_layers', 1), cheb_k=params.get('cheb_k', 2), output_dim=params.get('output_dim', 1))
    )


SPEC = ModelSpec(
    name='AGCRN',
    module='moderntsf.models.agcrn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/AGCRN.toml',
    model_card='src/moderntsf/models/agcrn/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)

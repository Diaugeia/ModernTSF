"""Model specification for DGCRN."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.dgcrn.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    """Validated DGCRN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``run_one.py``) and are therefore not declared here.
    """

    enc_in: int = Field(ge=1)
    gcn_depth: int = Field(default=1, ge=1)
    rnn_size: int = Field(default=16, ge=1)
    node_dim: int = Field(default=8, ge=1)
    hyper_gnn_dim: int = Field(default=8, ge=1)
    middle_dim: int = Field(default=2, ge=1)
    tanhalpha: float = Field(default=3.0, gt=0)
    dropout: float = Field(default=0.3, ge=0, lt=1)


def build_model(cfg, params):
    """Construct DGCRN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), gcn_depth=params.get('gcn_depth', 1), rnn_size=params.get('rnn_size', 16), node_dim=params.get('node_dim', 8), hyper_gnn_dim=params.get('hyper_gnn_dim', 8), middle_dim=params.get('middle_dim', 2), tanhalpha=params.get('tanhalpha', 3.0), dropout=params.get('dropout', 0.3))
    )


SPEC = ModelSpec(
    name='DGCRN',
    module='tsflab.models.dgcrn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/DGCRN.toml',
    model_card='src/tsflab/models/dgcrn/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)

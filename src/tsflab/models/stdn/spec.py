"""Model specification for STDN."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stdn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STDN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (not declared in TOML). ``enc_in`` (= ``N``) is the required node count and
    is used as the ``num_nodes`` fallback.
    """

    enc_in: int
    time_slice_size: int = 60
    K: int = 4
    d: int = 8
    L: int = 1
    order: int = 2
    reference: int = 4
    out_channels: int = 1


def build_model(cfg, params):
    """Construct STDN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), time_slice_size=params.get('time_slice_size', 60), K=params.get('K', 4), d=params.get('d', 8), L=params.get('L', 1), order=params.get('order', 2), reference=params.get('reference', 4), out_channels=params.get('out_channels', 1))
    )


SPEC = ModelSpec(
    name='STDN',
    module='tsflab.models.stdn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/STDN.toml',
    model_card='src/tsflab/models/stdn/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)

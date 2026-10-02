"""Model specification for STID."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stid.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STID parameters supplied via ``model.params``.

    ``enc_in`` (= number of nodes ``N``) is required. ``num_nodes`` and
    ``adj_mx`` are injected by the runner from the dataset, so they are not
    declared here. Defaults are modest for fast smoke runs.
    """

    enc_in: int
    input_dim: int = 3
    embed_dim: int = 32
    num_layers: int = 1
    num_time_in_day: int = 24
    num_day_in_week: int = 7
    if_time_in_day: bool = True
    if_day_in_week: bool = True


def build_model(cfg, params):
    """Construct STID from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3), embed_dim=params.get('embed_dim', 32), num_layers=params.get('num_layers', 1), num_time_in_day=params.get('num_time_in_day', 24), num_day_in_week=params.get('num_day_in_week', 7), if_time_in_day=params.get('if_time_in_day', True), if_day_in_week=params.get('if_day_in_week', True))
    )


SPEC = ModelSpec(
    name='STID',
    module='tsflab.models.stid',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/STID.toml',
    model_card='src/tsflab/models/stid/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)

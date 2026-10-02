"""Model specification for ExtraTreesTS."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.extra_trees_ts.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    num_estimators: int = Field(default=24, gt=0)
    tree_depth: int = Field(default=2, gt=0)
    threshold_range: float = Field(default=1.0, gt=0)
    temperature: float = Field(default=1.0, gt=0)
    random_seed: int = 1733
    use_revin: bool = True


def build_model(cfg, params):
    """Construct ExtraTreesTS from a validated run configuration."""
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len,
                 enc_in=params['enc_in'], num_estimators=params.get('num_estimators', 24),
                 tree_depth=params.get('tree_depth', 2),
                 threshold_range=params.get('threshold_range', 1.0),
                 temperature=params.get('temperature', 1.0),
                 random_seed=params.get('random_seed', 1733),
                 use_revin=bool(params.get('use_revin', True)))


SPEC = ModelSpec(
    name='ExtraTreesTS',
    module='tsflab.models.extra_trees_ts',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/ExtraTreesTS.toml',
    model_card='src/tsflab/models/extra_trees_ts/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
        components=('revin', 'soft_tree'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)

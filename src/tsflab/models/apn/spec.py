"""Model specification for APN."""
from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.apn.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    d_time: int = Field(default=8, gt=0)
    num_patches: int = Field(default=8, gt=0)


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params['enc_in'],
                 params.get('d_model', 64), params.get('d_time', 8),
                 params.get('num_patches', 8))


SPEC = ModelSpec(name='APN', module='tsflab.models.apn', model_class=Model,
                 factory=build_model, params_schema=ModelParameterConfig,
                 config_path='configs/models/APN.toml', model_card='src/tsflab/models/apn/README.md',
                 smoke_config=None, capabilities=frozenset(['time-series']),
                 components=(),
                 contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0})

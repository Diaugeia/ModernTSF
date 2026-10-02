"""Model specification for HN_MVTS."""
from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.hn_mvts.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    embedding_dim: int = Field(default=8, gt=0)
    hyper_hidden: int = Field(default=32, gt=0)
    use_revin: bool = True


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params['enc_in'],
                 params.get('d_model', 64), params.get('embedding_dim', 8),
                 params.get('hyper_hidden', 32), bool(params.get('use_revin', True)))


SPEC = ModelSpec(name='HN_MVTS', module='tsflab.models.hn_mvts', model_class=Model,
                 factory=build_model, params_schema=ModelParameterConfig,
                 config_path='configs/models/HN_MVTS.toml', model_card='src/tsflab/models/hn_mvts/README.md',
                 smoke_config=None, capabilities=frozenset(['time-series']),                  components=('revin',), contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0})

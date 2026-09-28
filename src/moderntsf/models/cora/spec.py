"""Model specification for CoRA."""
from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.cora.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    rank: int = Field(default=4, gt=0)
    polynomial_order: int = Field(default=2, ge=0)
    use_revin: bool = True


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params['enc_in'],
                 params.get('d_model', 64), params.get('rank', 4),
                 params.get('polynomial_order', 2), bool(params.get('use_revin', True)))


SPEC = ModelSpec(name='CoRA', module='moderntsf.models.cora', model_class=Model,
                 factory=build_model, params_schema=ModelParameterConfig,
                 config_path='configs/models/CoRA.toml', model_card='src/moderntsf/models/cora/README.md',
                 smoke_config=None, capabilities=frozenset(['time-series']),                  components=('revin',), contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0})

"""Runtime specification for ElasticNetTS."""

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.elastic_net_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    penalty: float = 1e-4
    l1_ratio: float = 0.5


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params["enc_in"], params.get("penalty", 1e-4), params.get("l1_ratio", 0.5))


SPEC = ModelSpec(name="ElasticNetTS", module="moderntsf.models.elastic_net_ts", model_class=Model, factory=build_model, params_schema=ModelParameterConfig, config_path="configs/models/ElasticNetTS.toml", model_card="src/moderntsf/models/elastic_net_ts/README.md", capabilities=frozenset(["time-series"]), components=(), contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0})

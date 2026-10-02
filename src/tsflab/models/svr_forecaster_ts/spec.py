"""Runtime specification for SVRForecasterTS."""

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.svr_forecaster_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    num_support: int = 16
    kernel_gamma: float = 0.1
    epsilon: float = 0.1
    l2_penalty: float = 1e-4


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params["enc_in"], params.get("num_support", 16), params.get("kernel_gamma", 0.1), params.get("epsilon", 0.1), params.get("l2_penalty", 1e-4))


SPEC = ModelSpec(name="SVRForecasterTS", module="tsflab.models.svr_forecaster_ts", model_class=Model, factory=build_model, params_schema=ModelParameterConfig, config_path="configs/models/SVRForecasterTS.toml", model_card="src/tsflab/models/svr_forecaster_ts/README.md", capabilities=frozenset(["time-series"]), components=(), contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0})

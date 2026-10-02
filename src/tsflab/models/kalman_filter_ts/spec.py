"""Runtime specification for KalmanFilterTS."""

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.kalman_filter_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    initial_alpha: float = 0.5
    initial_beta: float = 0.25


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params["enc_in"], params.get("initial_alpha", 0.5), params.get("initial_beta", 0.25))


SPEC = ModelSpec(name="KalmanFilterTS", module="tsflab.models.kalman_filter_ts", model_class=Model, factory=build_model, params_schema=ModelParameterConfig, config_path="configs/models/KalmanFilterTS.toml", model_card="src/tsflab/models/kalman_filter_ts/README.md", capabilities=frozenset(["time-series"]), components=(), contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0})

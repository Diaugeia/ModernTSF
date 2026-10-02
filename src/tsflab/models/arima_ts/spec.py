"""Runtime specification for ARIMATS."""

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.arima_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    ar_order: int = 2
    ma_order: int = 1


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params["enc_in"], params.get("ar_order", 2), params.get("ma_order", 1))


SPEC = ModelSpec(name="ARIMATS", module="tsflab.models.arima_ts", model_class=Model, factory=build_model, params_schema=ModelParameterConfig, config_path="configs/models/ARIMATS.toml", model_card="src/tsflab/models/arima_ts/README.md", capabilities=frozenset(["time-series"]), components=(), contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0})

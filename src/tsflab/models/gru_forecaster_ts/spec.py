"""Runtime specification for GRUForecasterTS."""

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.gru_forecaster_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    num_layers: int = 1
    use_revin: bool = True


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="GRUForecasterTS", module="tsflab.models.gru_forecaster_ts", model_class=Model,
    factory=build_model, params_schema=ModelParameterConfig,
    config_path="configs/models/GRUForecasterTS.toml",
    model_card="src/tsflab/models/gru_forecaster_ts/README.md",
    capabilities=frozenset(["time-series"]), components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)

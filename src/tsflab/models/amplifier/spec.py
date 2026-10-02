"""Runtime specification for Amplifier."""

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.amplifier.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    hidden_size: int = Field(default=128, gt=0)
    sci: bool = True
    moving_average: int = Field(default=25, gt=0)


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        hidden_size=params.get("hidden_size", 128),
        sci=params.get("sci", True),
        moving_average=params.get("moving_average", 25),
    )


SPEC = ModelSpec(
    name="Amplifier",
    module="tsflab.models.amplifier",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/Amplifier.toml",
    model_card="src/tsflab/models/amplifier/README.md",
    smoke_config=None,
    capabilities=frozenset(["time-series"]),
    components=("revin", "series_decomposition"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)

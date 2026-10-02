"""Runtime specification for TimeMachine."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.timemachine.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    n1: int = Field(default=256, ge=1)
    n2: int = Field(default=128, ge=1)
    d_state: int = Field(default=256, ge=1)
    d_conv: int = Field(default=2, ge=1)
    expand: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    revin: bool = True
    ch_ind: bool = True
    residual: bool = True


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        n1=params.get("n1", 256),
        n2=params.get("n2", 128),
        d_state=params.get("d_state", 256),
        d_conv=params.get("d_conv", 2),
        expand=params.get("expand", 1),
        dropout=params.get("dropout", 0.05),
        revin=params.get("revin", True),
        ch_ind=params.get("ch_ind", True),
        residual=params.get("residual", True),
    )


SPEC = ModelSpec(
    name="TimeMachine",
    module="tsflab.models.timemachine",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/TimeMachine.toml",
    model_card="src/tsflab/models/timemachine/README.md",
    smoke_config="configs/runs/smoke_timemachine.toml",
    capabilities=frozenset({"time-series"}),
    components=("mamba", "revin"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)

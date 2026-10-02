"""Runtime specification for SAMBA."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.samba.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=128, gt=0)
    d_ff: int = Field(default=128, gt=0)
    e_layers: int = Field(default=1, ge=0)
    d_layers: int = Field(default=1, ge=0)
    d_state1: int = Field(default=16, gt=0)
    d_state2: int = Field(default=16, gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    use_act: bool = False

    @model_validator(mode="after")
    def _at_least_one_encoder(self) -> "ModelParameterConfig":
        if self.e_layers + self.d_layers == 0:
            raise ValueError("at least one of e_layers and d_layers must be positive")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 128),
        d_ff=params.get("d_ff", 128),
        e_layers=params.get("e_layers", 1),
        d_layers=params.get("d_layers", 1),
        d_state1=params.get("d_state1", 16),
        d_state2=params.get("d_state2", 16),
        patch_len=params.get("patch_len", 16),
        stride=params.get("stride", 8),
        dropout=params.get("dropout", 0.1),
        use_act=bool(params.get("use_act", False)),
    )


SPEC = ModelSpec(
    name="SAMBA",
    module="tsflab.models.samba",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/SAMBA.toml",
    model_card="src/tsflab/models/samba/README.md",
    smoke_config="configs/runs/smoke_samba.toml",
    capabilities=frozenset({"time-series"}),
    components=("flatten_forecast_head", "mamba", "revin"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)

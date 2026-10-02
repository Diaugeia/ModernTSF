"""Model specification for FTP."""

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.ftp.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    num_layers: int = Field(default=2, gt=0)
    patch_unit: int = Field(default=4, gt=0)
    num_scales: int = Field(default=3, gt=0)
    stride: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    use_revin: bool = True


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        params.get("d_model", 64),
        params.get("num_layers", 2),
        params.get("patch_unit", 4),
        params.get("num_scales", 3),
        params.get("stride", 2),
        params.get("dropout", 0.1),
        bool(params.get("use_revin", True)),
    )


SPEC = ModelSpec(
    name="FTP",
    module="tsflab.models.ftp",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/FTP.toml",
    model_card="src/tsflab/models/ftp/README.md",
    capabilities=frozenset(["time-series"]),
        components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)

"""Runtime specification for SRSNet."""
from pydantic import BaseModel, Field
from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.srsnet.model import Model

class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=128, gt=0)
    patch_len: int = Field(default=24, gt=0)
    stride: int = Field(default=24, gt=0)
    hidden_size: int = Field(default=64, gt=0)
    dropout: float = Field(default=0.2, ge=0, lt=1)
    head_dropout: float = Field(default=0.1, ge=0, lt=1)
    alpha: float = Field(default=2.0, gt=0)
    pos: bool = True
    head_mode: str = Field(default="linear", pattern="^linear$")
    affine: bool = True
    subtract_last: bool = False

def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params["enc_in"],
        features=cfg.task.features, d_model=params.get("d_model",128),
        patch_len=params.get("patch_len",24), stride=params.get("stride",24),
        hidden_size=params.get("hidden_size",64), dropout=params.get("dropout",0.2),
        head_dropout=params.get("head_dropout",0.1), alpha=params.get("alpha",2.0),
        pos=params.get("pos",True), head_mode=params.get("head_mode","linear"),
        affine=params.get("affine",True), subtract_last=params.get("subtract_last",False))

SPEC = ModelSpec(name="SRSNet", module="tsflab.models.srsnet", model_class=Model,
    factory=build_model, params_schema=ModelParameterConfig,
    config_path="configs/models/SRSNet.toml", model_card="src/tsflab/models/srsnet/README.md",
    smoke_config=None, capabilities=frozenset(["time-series"]),
    components=("flatten_forecast_head","revin"),
    contract_task={"seq_len":96,"pred_len":96,"label_len":0})

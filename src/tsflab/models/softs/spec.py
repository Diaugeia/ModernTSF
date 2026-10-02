"""Runtime specification for SOFTS."""
from pydantic import BaseModel, Field
from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.softs.model import Model

class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=128, gt=0)
    d_core: int = Field(default=64, gt=0)
    d_ff: int = Field(default=256, gt=0)
    e_layers: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    activation: str = Field(default="gelu", pattern="^(gelu|relu)$")
    use_norm: bool = True

def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params["enc_in"],
        features=cfg.task.features, label_len=cfg.task.label_len,
        d_model=params.get("d_model",128), d_core=params.get("d_core",64),
        d_ff=params.get("d_ff",256), e_layers=params.get("e_layers",2),
        dropout=params.get("dropout",0.1), activation=params.get("activation","gelu"),
        use_norm=params.get("use_norm",True))

SPEC = ModelSpec(name="SOFTS", module="tsflab.models.softs", model_class=Model,
    factory=build_model, params_schema=ModelParameterConfig,
    config_path="configs/models/SOFTS.toml", model_card="src/tsflab/models/softs/README.md",
    smoke_config=None, capabilities=frozenset(["time-series"]), components=(),
    contract_task={"seq_len":96,"pred_len":96,"label_len":0})

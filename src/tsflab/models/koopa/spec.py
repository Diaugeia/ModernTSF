"""Runtime specification for Koopa."""
from pydantic import BaseModel, Field
from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.koopa.model import Model

class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    seg_len: int | None = Field(default=None, gt=1)
    dynamic_dim: int = Field(default=128, gt=0)
    hidden_dim: int = Field(default=64, gt=0)
    hidden_layers: int = Field(default=2, gt=0)
    num_blocks: int = Field(default=3, gt=0)
    multistep: bool = False
    alpha: float = Field(default=0.2, gt=0, le=1)

def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params["enc_in"],
        label_len=cfg.task.label_len, features=cfg.task.features,
        seg_len=params.get("seg_len"), dynamic_dim=params.get("dynamic_dim",128),
        hidden_dim=params.get("hidden_dim",64), hidden_layers=params.get("hidden_layers",2),
        num_blocks=params.get("num_blocks",3), multistep=params.get("multistep",False),
        alpha=params.get("alpha",0.2))

SPEC = ModelSpec(name="Koopa", module="tsflab.models.koopa", model_class=Model,
    factory=build_model, params_schema=ModelParameterConfig,
    config_path="configs/models/Koopa.toml", model_card="src/tsflab/models/koopa/README.md",
    smoke_config=None, capabilities=frozenset(["time-series"]), components=(),
    contract_task={"seq_len":192,"pred_len":96,"label_len":0})

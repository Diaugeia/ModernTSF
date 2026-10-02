"""Runtime specification for TimeAlign."""
from pydantic import BaseModel, Field
from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.timealign.model import Model

class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    patch_num: int = Field(default=4, gt=0)
    d_model: int = Field(default=32, gt=0)
    d_ff: int = Field(default=32, gt=0)
    e_layers: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    pos: bool = True
    layer_norm: bool = True
    loc: bool = True
    glo: bool = True
    local_margin: float = Field(default=0.0, ge=0)
    global_margin: float = Field(default=0.0, ge=0)
    w_recon: float = Field(default=1.0, ge=0)
    w_align: float = Field(default=0.1, ge=0)

def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params["enc_in"],
        patch_num=params.get("patch_num",4), d_model=params.get("d_model",32),
        d_ff=params.get("d_ff",32), e_layers=params.get("e_layers",2),
        dropout=params.get("dropout",0.1), pos=params.get("pos",True),
        layer_norm=params.get("layer_norm",True), loc=params.get("loc",True),
        glo=params.get("glo",True), local_margin=params.get("local_margin",0.0),
        global_margin=params.get("global_margin",0.0), w_recon=params.get("w_recon",1.0),
        w_align=params.get("w_align",0.1))

def training_objective(model, batch_x, target):
    forecast, loss, _ = model.training_objective(batch_x, target)
    return forecast, loss

SPEC = ModelSpec(name="TimeAlign", module="tsflab.models.timealign", model_class=Model,
    factory=build_model, params_schema=ModelParameterConfig,
    config_path="configs/models/TimeAlign.toml", model_card="src/tsflab/models/timealign/README.md",
    smoke_config="configs/runs/smoke_timealign.toml", capabilities=frozenset(["time-series"]),
    components=("revin",), contract_task={"seq_len":96,"pred_len":96,"label_len":0},
    training_objective=training_objective)

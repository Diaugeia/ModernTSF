"""Runtime specification for LatentTSF."""
from pydantic import BaseModel, Field
from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.latenttsf.model import Model

class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    d_ff: int = Field(default=128, gt=0)
    mse_weight: float = Field(default=10.0, ge=0)
    cosine_weight: float = Field(default=15.0, ge=0)
    use_latent_norm: bool = True
    kernel_size: int = Field(default=25, gt=0)
    individual: bool = False
    ae_train_epochs: int = Field(default=100, ge=0)
    ae_lr: float = Field(default=5e-4, gt=0)
    ae_loss: str = Field(default="MAE", pattern="^(MAE|MSE)$")

def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params["enc_in"],
        d_model=params.get("d_model",64), d_ff=params.get("d_ff",128),
        mse_weight=params.get("mse_weight",10.0), cosine_weight=params.get("cosine_weight",15.0),
        use_latent_norm=params.get("use_latent_norm",True), kernel_size=params.get("kernel_size",25),
        individual=params.get("individual",False), ae_train_epochs=params.get("ae_train_epochs",100),
        ae_lr=params.get("ae_lr",5e-4), ae_loss=params.get("ae_loss","MAE"))

def training_objective(model, batch_x, target):
    forecast, loss, _ = model.training_objective(batch_x, target)
    return forecast, loss

SPEC = ModelSpec(name="LatentTSF", module="moderntsf.models.latenttsf", model_class=Model,
    factory=build_model, params_schema=ModelParameterConfig,
    config_path="configs/models/LatentTSF.toml", model_card="src/moderntsf/models/latenttsf/README.md",
    smoke_config="configs/runs/smoke_latenttsf.toml", capabilities=frozenset(["time-series", "pretraining-stage"]),
    components=("dlinear",), contract_task={"seq_len":96,"pred_len":96,"label_len":0},
    training_objective=training_objective)

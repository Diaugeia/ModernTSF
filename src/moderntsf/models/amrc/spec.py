"""Model specification for AMRC."""

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.amrc.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    mask_samples: int = Field(default=4, gt=0)
    lambda_aml: float = Field(default=0.1, ge=0)
    lambda_esp: float = Field(default=0.1, ge=0)
    use_revin: bool = True


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        params.get("d_model", 64),
        params.get("mask_samples", 4),
        params.get("lambda_aml", 0.1),
        params.get("lambda_esp", 0.1),
        bool(params.get("use_revin", True)),
    )


def training_objective(model, batch_x, target):
    forecast, loss, _ = model.training_objective(batch_x, target)
    return forecast, loss


SPEC = ModelSpec(
    name="AMRC",
    module="moderntsf.models.amrc",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/AMRC.toml",
    model_card="src/moderntsf/models/amrc/README.md",
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)

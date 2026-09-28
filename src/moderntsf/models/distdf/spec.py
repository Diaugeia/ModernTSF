"""Model specification for DistDF."""

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.distdf.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    gamma: float = Field(default=0.1, ge=0, le=1)
    covariance_eps: float = Field(default=1e-5, gt=0)
    use_revin: bool = True


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        params.get("gamma", 0.1),
        params.get("covariance_eps", 1e-5),
        bool(params.get("use_revin", True)),
    )


def training_objective(model, batch_x, target):
    forecast, loss, _ = model.training_objective(batch_x, target)
    return forecast, loss


SPEC = ModelSpec(
    name="DistDF",
    module="moderntsf.models.distdf",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/DistDF.toml",
    model_card="src/moderntsf/models/distdf/README.md",
    capabilities=frozenset(["time-series"]),
    components=("channel_wise_linear", "revin"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)

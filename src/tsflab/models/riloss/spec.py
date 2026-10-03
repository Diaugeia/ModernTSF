"""Runtime specification for RILoss."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.riloss.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    kernel_size: int = Field(default=25, ge=1)
    individual: bool = False
    ri_weight: float = Field(default=10.0, ge=0.0)
    temperature: float = Field(default=1.0, ge=0.0)
    bandwidth: float = Field(default=1.0, gt=0.0)
    noise_low: float = 0.0
    noise_high: float = 1.0

    @model_validator(mode="after")
    def _check(self):
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if self.noise_high <= self.noise_low:
            raise ValueError("noise_high must exceed noise_low")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        kernel_size=params.get("kernel_size", 25),
        individual=params.get("individual", False),
        ri_weight=params.get("ri_weight", 10.0),
        temperature=params.get("temperature", 1.0),
        bandwidth=params.get("bandwidth", 1.0),
        noise_low=params.get("noise_low", 0.0),
        noise_high=params.get("noise_high", 1.0),
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """RI-Loss (Eq. 8): the configured observation loss plus the HSIC residual term."""
    forecast = batch.forecast(model)
    aligned, target = batch.align(forecast), batch.target
    if target.shape[-1] < 2:
        raise ValueError(
            "RI-Loss measures HSIC across the channels of each window; it needs at least "
            "two target channels (use features = 'M')"
        )
    return forecast, model.ri_loss(aligned, target, criterion(aligned, target))


SPEC = ModelSpec(
    name="RILoss",
    module="tsflab.models.riloss",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/RILoss.toml",
    model_card="src/tsflab/models/riloss/README.md",
    smoke_config="configs/runs/smoke_riloss.toml",
    capabilities=frozenset({"time-series"}),
    components=("dlinear",),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
)

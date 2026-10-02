"""Model specification for Aurora."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.aurora.model import Model
from pydantic import BaseModel, Field, model_validator


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    patch_len: int = Field(default=16, gt=0)
    num_heads: int = Field(default=4, gt=0)
    num_distill_tokens: int = Field(default=2, gt=0)
    num_prototypes: int = Field(default=8, gt=0)
    flow_steps: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    use_revin: bool = True

    @model_validator(mode="after")
    def validate_heads(self):
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        return self


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        params.get("d_model", 64),
        params.get("patch_len", 16),
        params.get("num_heads", 4),
        params.get("num_distill_tokens", 2),
        params.get("num_prototypes", 8),
        params.get("flow_steps", 2),
        params.get("dropout", 0.1),
        bool(params.get("use_revin", True)),
    )


SPEC = ModelSpec(
    name="Aurora",
    module="tsflab.models.aurora",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/Aurora.toml",
    model_card="src/tsflab/models/aurora/README.md",
    capabilities=frozenset(["time-series", "dense-modality-context"]),
        components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)

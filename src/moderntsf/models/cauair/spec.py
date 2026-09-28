"""Runtime specification for CauAir."""
from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.cauair.model import Model
from pydantic import BaseModel, Field, model_validator


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    cov_dim: int = Field(default=2, gt=0)
    dim: int = Field(default=64, gt=0)
    cache_count: int = Field(default=8, gt=0)
    heads: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def validate_heads(self):
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        return self


def build_model(cfg, params):
    params.pop("adj_mx", None)
    params.pop("num_nodes", None)
    return Model(cfg.task.seq_len, cfg.task.pred_len, **params)


SPEC = ModelSpec(name="CauAir", module="moderntsf.models.cauair", model_class=Model,
    factory=build_model, params_schema=ModelParameterConfig,
    config_path="configs/models/CauAir.toml", model_card="src/moderntsf/models/cauair/README.md",
    capabilities=frozenset(["covariate"]), components=("marks",),
    contract_task={"seq_len": 24, "pred_len": 24, "label_len": 0})

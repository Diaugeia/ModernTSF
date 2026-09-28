"""Runtime specification for DeepAir."""
from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.deepair.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    cov_dim: int = Field(default=2, gt=0)
    hidden_dim: int = Field(default=32, gt=0)
    spatial_regions: int = Field(default=4, gt=0)


def build_model(cfg, params):
    params.pop("num_nodes", None)
    return Model(cfg.task.seq_len, cfg.task.pred_len,
        spatial_mx=params.pop("adj_mx", None), **params)


SPEC = ModelSpec(name="DeepAir", module="moderntsf.models.deepair", model_class=Model,
    factory=build_model, params_schema=ModelParameterConfig,
    config_path="configs/models/DeepAir.toml", model_card="src/moderntsf/models/deepair/README.md",
    capabilities=frozenset(["covariate"]), components=("marks",),
    contract_task={"seq_len": 24, "pred_len": 24, "label_len": 0})

"""Runtime specification for Sumba."""
from pydantic import BaseModel, Field
from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.sumba.model import Model

class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=32, gt=0)
    basis_count: int = Field(default=4, gt=0)
    basis_rank: int = Field(default=8, gt=0)
    temporal_kernels: list[int] = Field(default=[2,3,5], min_length=1)
    depth: int = Field(default=2, gt=0)
    diffusion_steps: int = Field(default=2, gt=0)
    mix: float = Field(default=0.1, ge=0, le=1)
    dropout: float = Field(default=0.1, ge=0, lt=1)

def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params["enc_in"],
        label_len=cfg.task.label_len, features=cfg.task.features,
        d_model=params.get("d_model",32), basis_count=params.get("basis_count",4),
        basis_rank=params.get("basis_rank",8),
        temporal_kernels=tuple(params.get("temporal_kernels",(2,3,5))),
        depth=params.get("depth",2), diffusion_steps=params.get("diffusion_steps",2),
        mix=params.get("mix",0.1), dropout=params.get("dropout",0.1))

SPEC = ModelSpec(name="Sumba", module="tsflab.models.sumba", model_class=Model,
    factory=build_model, params_schema=ModelParameterConfig,
    config_path="configs/models/Sumba.toml", model_card="src/tsflab/models/sumba/README.md",
    smoke_config=None, capabilities=frozenset(["time-series"]), components=(),
    contract_task={"seq_len":96,"pred_len":96,"label_len":0})

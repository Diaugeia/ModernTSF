"""Runtime specification for PULSE."""

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.pulse.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 32
    phase_period: int = 24
    phase_resolution: int = 8
    router_heads: int = 4
    dropout: float = 0.1
    eps: float = 1e-5


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 32),
        phase_period=params.get("phase_period", 24),
        phase_resolution=params.get("phase_resolution", 8),
        router_heads=params.get("router_heads", 4),
        dropout=params.get("dropout", 0.1),
        eps=params.get("eps", 1e-5),
    )


SPEC = ModelSpec(
    name="PULSE",
    module="moderntsf.models.pulse",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/PULSE.toml",
    model_card="src/moderntsf/models/pulse/README.md",
    smoke_config=None,
    capabilities=frozenset(["time-series"]),
        components=(),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)

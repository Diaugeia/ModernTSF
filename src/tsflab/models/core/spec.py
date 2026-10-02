"""Runtime specification for CoRe."""

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.core.model import Model

"""Validated parameters for CoRe."""

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    context_len: int = 5
    var_wise_gating: bool = True
    scr_rank: int = 0
    gate_init: float = 0.0
    gate_bias_init: float = -1.0


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        context_len=params.get("context_len", 5),
        var_wise_gating=params.get("var_wise_gating", True),
        scr_rank=params.get("scr_rank", 0),
        gate_init=params.get("gate_init", 0.0),
        gate_bias_init=params.get("gate_bias_init", -1.0),
    )


SPEC = ModelSpec(
    name="CoRe",
    module="tsflab.models.core",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/CoRe.toml",
    model_card="src/tsflab/models/core/README.md",
    smoke_config="configs/runs/smoke_core.toml",
    capabilities=frozenset({'time-series', 'test-time-adaptation'}),
    components=('channel_wise_linear', 'spectral_descriptor'),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)

"""Runtime specification for FreqMoE."""

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.freqmoe.model import Model

"""Validated parameters for FreqMoE."""

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    expert_num: int = 4
    freq_num_blocks: int = 1
    dropout_freq: float = 0.1


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        expert_num=params.get("expert_num", 4),
        freq_num_blocks=params.get("freq_num_blocks", 1),
        dropout_freq=params.get("dropout_freq", 0.1),
    )


SPEC = ModelSpec(
    name="FreqMoE",
    module="moderntsf.models.freqmoe",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/FreqMoE.toml",
    model_card="src/moderntsf/models/freqmoe/README.md",
    smoke_config="configs/runs/smoke_freqmoe.toml",
    capabilities=frozenset({'time-series'}),
    components=('freq_band_moe',),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)

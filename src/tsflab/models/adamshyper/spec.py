"""Runtime specification for AdaMSHyper."""

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.adamshyper.model import Model

"""Validated parameters for AdaMSHyper."""

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    window_size: list[int] = [4, 4]
    hyper_num: list[int] = [50, 20, 10]
    d_embed: int = 16
    eta: int = 3
    beta: float = 0.5
    gamma: float = 4.2
    lambda_balance: float = 0.5
    const_weight: float = 1.0
    dropout: float = 0.1


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        window_size=params.get("window_size", [4, 4]),
        hyper_num=params.get("hyper_num", [50, 20, 10]),
        d_embed=params.get("d_embed", 16),
        eta=params.get("eta", 3),
        beta=params.get("beta", 0.5),
        gamma=params.get("gamma", 4.2),
        lambda_balance=params.get("lambda_balance", 0.5),
        const_weight=params.get("const_weight", 1.0),
        dropout=params.get("dropout", 0.1),
    )


SPEC = ModelSpec(
    name="AdaMSHyper",
    module="tsflab.models.adamshyper",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/AdaMSHyper.toml",
    model_card="src/tsflab/models/adamshyper/README.md",
    smoke_config="configs/runs/smoke_adamshyper.toml",
    capabilities=frozenset({'time-series'}),
    components=('adaptive_node_embedding_adjacency', 'revin'),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)

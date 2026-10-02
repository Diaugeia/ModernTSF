"""Model specification for ST-SSDL."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.st_ssdl.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated ST-SSDL parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset,
    so they are not declared here. ``enc_in`` (= number of nodes ``N``) is
    required; the remaining defaults are kept modest for a fast spatiotemporal
    smoke run.
    """

    enc_in: int
    input_dim: int = 3
    rnn_units: int = 16
    rnn_layers: int = 1
    cheb_k: int = 2
    prototype_num: int = 8
    prototype_dim: int = 8
    output_dim: int = 1


def build_model(cfg, params):
    """Construct ST-SSDL from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        input_dim=params.get("input_dim", 3),
        rnn_units=params.get("rnn_units", 16),
        rnn_layers=params.get("rnn_layers", 1),
        cheb_k=params.get("cheb_k", 2),
        prototype_num=params.get("prototype_num", 8),
        prototype_dim=params.get("prototype_dim", 8),
        output_dim=params.get("output_dim", 1),
    )


SPEC = ModelSpec(
    name="ST-SSDL",
    module="tsflab.models.st_ssdl",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/ST-SSDL.toml",
    model_card="src/tsflab/models/st_ssdl/README.md",
    smoke_config=None,
    capabilities=frozenset(["spatiotemporal"]),
    components=("deviation_memory", "graph_utils", "marks"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)

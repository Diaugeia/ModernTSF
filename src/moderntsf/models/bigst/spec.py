"""Model specification for BigST."""

from __future__ import annotations

from moderntsf.benchmark.registry.models import ModelSpec
from moderntsf.models.bigst.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated BigST parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    and are not declared here.
    """

    enc_in: int  # number of nodes N (required)
    input_dim: int = 3  # 1 value + calendar covariates [tod, dow]
    hid_dim: int = 16
    node_dim: int = 8
    time_dim: int = 8
    tod_size: int = 24
    dow_size: int = 7
    tau: float = 1.0
    random_feature_dim: int = 16
    dropout: float = 0.1
    use_residual: bool = True
    use_bn: bool = True


def build_model(cfg, params):
    """Construct BigST from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3), hid_dim=params.get('hid_dim', 16), node_dim=params.get('node_dim', 8), time_dim=params.get('time_dim', 8), tod_size=params.get('tod_size', 24), dow_size=params.get('dow_size', 7), tau=params.get('tau', 1.0), random_feature_dim=params.get('random_feature_dim', 16), dropout=params.get('dropout', 0.1), use_residual=params.get('use_residual', True), use_bn=params.get('use_bn', True))
    )


SPEC = ModelSpec(
    name='BigST',
    module='moderntsf.models.bigst',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/BigST.toml',
    model_card='src/moderntsf/models/bigst/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)

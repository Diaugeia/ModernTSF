"""Model specification for MTGNN."""

from __future__ import annotations

from tsflab.benchmark.registry.models import ModelSpec
from tsflab.models.mtgnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated MTGNN parameters supplied via ``model.params``.

    ``enc_in`` (= number of nodes ``N``) is required. ``num_nodes`` and
    ``adj_mx`` are injected by the runner from the dataset and need not be
    declared in the TOML.
    """

    enc_in: int
    input_dim: int = 3
    gcn_depth: int = 2
    subgraph_size: int = 20
    node_dim: int = 40
    conv_channels: int = 16
    residual_channels: int = 16
    skip_channels: int = 32
    end_channels: int = 64
    layers: int = 3
    dropout: float = 0.3
    propalpha: float = 0.05
    tanhalpha: float = 3.0
    dilation_exponential: int = 1
    build_adj: bool = True


def build_model(cfg, params):
    """Construct MTGNN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3), gcn_depth=params.get('gcn_depth', 2), subgraph_size=params.get('subgraph_size', 20), node_dim=params.get('node_dim', 40), conv_channels=params.get('conv_channels', 16), residual_channels=params.get('residual_channels', 16), skip_channels=params.get('skip_channels', 32), end_channels=params.get('end_channels', 64), layers=params.get('layers', 3), dropout=params.get('dropout', 0.3), propalpha=params.get('propalpha', 0.05), tanhalpha=params.get('tanhalpha', 3.0), dilation_exponential=params.get('dilation_exponential', 1), build_adj=params.get('build_adj', True))
    )


SPEC = ModelSpec(
    name='MTGNN',
    module='tsflab.models.mtgnn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/MTGNN.toml',
    model_card='src/tsflab/models/mtgnn/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('gated_dilated_conv', 'marks'),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)

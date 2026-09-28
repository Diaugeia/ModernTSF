"""Flat, lazy model catalog and model specification contracts."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Literal, Type

from pydantic import BaseModel


@dataclass(frozen=True)
class ModelArtifact:
    """Pinned external artifact required by an optional model runtime path."""

    name: str
    url: str
    revision: str
    sha256: str
    filename: str
    required: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.revision or not self.filename:
            raise ValueError("artifact name, revision, and filename must be non-empty")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError(f"artifact {self.name!r} needs a lowercase SHA-256 digest")
        if not self.url.startswith(("https://", "file://")):
            raise ValueError(f"artifact {self.name!r} URL must use https:// or file://")
        if self.url.startswith("https://") and self.revision not in self.url:
            raise ValueError(
                f"artifact {self.name!r} URL must contain its pinned revision"
            )
        if "/" in self.filename or "\\" in self.filename:
            raise ValueError(f"artifact {self.name!r} filename must be a basename")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    module: str
    model_class: type
    factory: Callable
    params_schema: Type[BaseModel]
    config_path: str = ""
    model_card: str = ""
    smoke_config: str | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)
    components: tuple[str, ...] = ()
    contract_task: dict[str, int | str] = field(default_factory=dict)
    contract_seeds: tuple[int, ...] = (0,)
    artifacts: tuple[ModelArtifact, ...] = ()
    artifact_factory: Callable | None = None
    training_objective: Callable | None = None

    def __post_init__(self) -> None:
        output_capabilities = self.capabilities & {
            "quantile-output", "distribution-output"
        }
        if len(output_capabilities) > 1:
            raise ValueError(f"model {self.name!r} declares conflicting output capabilities")
        if not self.task_modes:
            raise ValueError(f"model {self.name!r} declares no supported task mode")
        if len(set(self.components)) != len(self.components):
            raise ValueError(f"model {self.name!r} declares duplicate components")
        if self.training_objective is not None and not callable(self.training_objective):
            raise TypeError(f"model {self.name!r} training_objective must be callable")
        if "inference-only" in self.capabilities:
            if "pretraining-stage" in self.capabilities:
                raise ValueError(
                    f"model {self.name!r} cannot be inference-only and pretraining-stage"
                )
            if self.training_objective is not None:
                raise ValueError(
                    f"model {self.name!r} cannot be inference-only with a training objective"
                )
        artifact_names = [artifact.name for artifact in self.artifacts]
        if len(set(artifact_names)) != len(artifact_names):
            raise ValueError(f"model {self.name!r} declares duplicate artifacts")
        if self.artifact_factory is not None and not callable(self.artifact_factory):
            raise TypeError(f"model {self.name!r} artifact_factory must be callable")
        if self.artifact_factory is not None and not self.artifacts:
            raise ValueError(
                f"model {self.name!r} declares artifact_factory without artifacts"
            )

    @property
    def output_type(self) -> Literal["point", "quantile", "distribution"]:
        """Declared public output kind, derived from orthogonal capabilities."""
        if "quantile-output" in self.capabilities:
            return "quantile"
        if "distribution-output" in self.capabilities:
            return "distribution"
        return "point"

    @property
    def task_modes(self) -> frozenset[str]:
        """Return supported public data settings from orthogonal capabilities."""
        mapping = {
            "time-series": "time_series",
            "spatiotemporal": "spatiotemporal",
            "covariate": "covariate",
        }
        return frozenset(mapping[key] for key in mapping if key in self.capabilities)

    def validate_params(self, params: dict) -> dict:
        unknown = sorted(set(params) - set(self.params_schema.model_fields))
        if unknown:
            raise ValueError(
                f"Unknown parameters for {self.name}: {', '.join(unknown)}"
            )
        return self.params_schema.model_validate(params).model_dump()

    def build(self, cfg, params: dict):
        return self.factory(cfg, self.validate_params(params))

    def build_with_artifacts(self, cfg, params: dict, paths: dict):
        """Construct an artifact-backed model through its explicit factory."""
        if self.artifact_factory is None:
            raise ValueError(
                f"model {self.name!r} declares runtime artifacts but has no artifact_factory"
            )
        return self.artifact_factory(cfg, self.validate_params(params), dict(paths))


class ModelCatalog:
    def __init__(self, refs: dict[str, str]) -> None:
        self._refs = dict(refs)
        self._loaded: dict[str, ModelSpec] = {}

    def names(self) -> list[str]:
        return sorted(self._refs)

    def refs(self) -> dict[str, str]:
        return dict(self._refs)

    def get(self, name: str) -> ModelSpec:
        if name in self._loaded:
            return self._loaded[name]
        module_path = self._refs.get(name)
        if module_path is None:
            available = ", ".join(self.names())
            raise KeyError(f"Unknown model {name!r}. Available: {available}")
        module = importlib.import_module(module_path)
        spec = getattr(module, "SPEC", None)
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"{module_path} must expose a ModelSpec named SPEC")
        if spec.name != name:
            raise ValueError(
                f"catalog key {name!r} disagrees with {module_path}.SPEC.name={spec.name!r}"
            )
        self._loaded[name] = spec
        return spec


MODEL_CATALOG = ModelCatalog({
    'BiMamba': 'moderntsf.models.bimamba.spec',
    'WPMixer': 'moderntsf.models.wpmixer.spec',
    'DLinear': 'moderntsf.models.dlinear.spec',
    'Linear': 'moderntsf.models.linear.spec',
    'NLinear': 'moderntsf.models.nlinear.spec',
    'RLinear': 'moderntsf.models.rlinear.spec',
    'CMoS': 'moderntsf.models.cmos.spec',
    'CycleNet': 'moderntsf.models.cyclenet.spec',
    'TimeEmb': 'moderntsf.models.timeemb.spec',
    'MixLinear': 'moderntsf.models.mixlinear.spec',
    'PWS': 'moderntsf.models.pws.spec',
    'PaiFilter': 'moderntsf.models.paifilter.spec',
    'FITS': 'moderntsf.models.fits.spec',
    'SVTime': 'moderntsf.models.svtime.spec',
    'SparseTSF': 'moderntsf.models.sparsetsf.spec',
    'TexFilter': 'moderntsf.models.texfilter.spec',
    'Autoformer': 'moderntsf.models.autoformer.spec',
    'FEDformer': 'moderntsf.models.fedformer.spec',
    'PatchTST': 'moderntsf.models.patchtst.spec',
    'PatchMLP': 'moderntsf.models.patchmlp.spec',
    'xPatch': 'moderntsf.models.xpatch.spec',
    'Amplifier': 'moderntsf.models.amplifier.spec',
    'CrossLinear': 'moderntsf.models.crosslinear.spec',
    'TimeBase': 'moderntsf.models.timebase.spec',
    'TimeBridge': 'moderntsf.models.timebridge.spec',
    'SegRNN': 'moderntsf.models.segrnn.spec',
    'TSMixer': 'moderntsf.models.tsmixer.spec',
    'LightTS': 'moderntsf.models.lightts.spec',
    'SCINet': 'moderntsf.models.scinet.spec',
    'TiDE': 'moderntsf.models.tide.spec',
    'TimeMixer': 'moderntsf.models.timemixer.spec',
    'TimesNet': 'moderntsf.models.timesnet.spec',
    'iTransformer': 'moderntsf.models.itransformer.spec',
    'STNorm': 'moderntsf.models.stnorm.spec',
    'TimeXer': 'moderntsf.models.timexer.spec',
    'TimeFilter': 'moderntsf.models.timefilter.spec',
    'MambaSimple': 'moderntsf.models.mambasimple.spec',
    'S_Mamba': 'moderntsf.models.s_mamba.spec',
    'S4': 'moderntsf.models.s4.spec',
    'MSGNet': 'moderntsf.models.msgnet.spec',
    'HDMixer': 'moderntsf.models.hdmixer.spec',
    'DSFormer': 'moderntsf.models.dsformer.spec',
    'UMixer': 'moderntsf.models.umixer.spec',
    'TimeKAN': 'moderntsf.models.timekan.spec',
    'Fredformer': 'moderntsf.models.fredformer.spec',
    'PAttn': 'moderntsf.models.pattn.spec',
    'CARD': 'moderntsf.models.card.spec',
    'NHiTS': 'moderntsf.models.nhits.spec',
    'NBeats': 'moderntsf.models.nbeats.spec',
    'DUET': 'moderntsf.models.duet.spec',
    'ETSformer': 'moderntsf.models.etsformer.spec',
    'NSTransformer': 'moderntsf.models.nstransformer.spec',
    'SOFTS': 'moderntsf.models.softs.spec',
    'Transformer': 'moderntsf.models.transformer.spec',
    'Reformer': 'moderntsf.models.reformer.spec',
    'Pyraformer': 'moderntsf.models.pyraformer.spec',
    'MultiPatchFormer': 'moderntsf.models.multipatchformer.spec',
    'ModernTCN': 'moderntsf.models.moderntcn.spec',
    'Crossformer': 'moderntsf.models.crossformer.spec',
    'FreTS': 'moderntsf.models.frets.spec',
    'FiLM': 'moderntsf.models.film.spec',
    'MICN': 'moderntsf.models.micn.spec',
    'Koopa': 'moderntsf.models.koopa.spec',
    'Informer': 'moderntsf.models.informer.spec',
    'MTSMixer': 'moderntsf.models.mtsmixer.spec',
    'Pathformer': 'moderntsf.models.pathformer.spec',
    'WaveNet': 'moderntsf.models.wavenet.spec',
    'DeepAR': 'moderntsf.models.deepar.spec',
    'Sumba': 'moderntsf.models.sumba.spec',
    'SRSNet': 'moderntsf.models.srsnet.spec',
    'DTAF': 'moderntsf.models.dtaf.spec',
    'TimePerceiver': 'moderntsf.models.timeperceiver.spec',
    'CrossGNN': 'moderntsf.models.crossgnn.spec',
    'RidgeRegressionTS': 'moderntsf.models.ridge_regression_ts.spec',
    'LassoRegressionTS': 'moderntsf.models.lasso_regression_ts.spec',
    'ElasticNetTS': 'moderntsf.models.elastic_net_ts.spec',
    'BayesianRidgeTS': 'moderntsf.models.bayesian_ridge_ts.spec',
    'PolynomialRegressionTS': 'moderntsf.models.polynomial_regression_ts.spec',
    'KNNForecasterTS': 'moderntsf.models.knn_forecaster_ts.spec',
    'SVRForecasterTS': 'moderntsf.models.svr_forecaster_ts.spec',
    'GaussianProcessTS': 'moderntsf.models.gaussian_process_ts.spec',
    'DecisionTreeTS': 'moderntsf.models.decision_tree_ts.spec',
    'RandomForestTS': 'moderntsf.models.random_forest_ts.spec',
    'ExtraTreesTS': 'moderntsf.models.extra_trees_ts.spec',
    'GradientBoostingTS': 'moderntsf.models.gradient_boosting_ts.spec',
    'XGBoostTS': 'moderntsf.models.xgboost_ts.spec',
    'LightGBMTS': 'moderntsf.models.lightgbm_ts.spec',
    'CatBoostTS': 'moderntsf.models.catboost_ts.spec',
    'ARIMATS': 'moderntsf.models.arima_ts.spec',
    'AutoRegressiveTS': 'moderntsf.models.autoregressive_ts.spec',
    'ExpSmoothingTS': 'moderntsf.models.exp_smoothing_ts.spec',
    'KalmanFilterTS': 'moderntsf.models.kalman_filter_ts.spec',
    'MLPForecasterTS': 'moderntsf.models.mlp_forecaster_ts.spec',
    'RNNForecasterTS': 'moderntsf.models.rnn_forecaster_ts.spec',
    'GRUForecasterTS': 'moderntsf.models.gru_forecaster_ts.spec',
    'LSTMForecasterTS': 'moderntsf.models.lstm_forecaster_ts.spec',
    'TCNForecasterTS': 'moderntsf.models.tcn_forecaster_ts.spec',
    'Aurora': 'moderntsf.models.aurora.spec',
    'CRIB': 'moderntsf.models.crib.spec',
    'TimeAlign': 'moderntsf.models.timealign.spec',
    'GTR': 'moderntsf.models.gtr.spec',
    'PhaseFormer': 'moderntsf.models.phaseformer.spec',
    'PMDformer': 'moderntsf.models.pmdformer.spec',
    'MMPD': 'moderntsf.models.mmpd.spec',
    'COSA': 'moderntsf.models.cosa.spec',
    'DistDF': 'moderntsf.models.distdf.spec',
    'Sonnet': 'moderntsf.models.sonnet.spec',
    'APN': 'moderntsf.models.apn.spec',
    'TimeCAP': 'moderntsf.models.timecap.spec',
    'GOTSF': 'moderntsf.models.gotsf.spec',
    'FTP': 'moderntsf.models.ftp.spec',
    'OccamVTS': 'moderntsf.models.occamvts.spec',
    'HN_MVTS': 'moderntsf.models.hn_mvts.spec',
    'SEMPO': 'moderntsf.models.sempo.spec',
    'InterPDN': 'moderntsf.models.interpdn.spec',
    'TimeO1': 'moderntsf.models.timeo1.spec',
    'FeTS': 'moderntsf.models.fets.spec',
    'SymTime': 'moderntsf.models.symtime.spec',
    'ImplicitForecaster': 'moderntsf.models.implicitforecaster.spec',
    'AMRC': 'moderntsf.models.amrc.spec',
    'HMformer': 'moderntsf.models.hmformer.spec',
    'TiRex': 'moderntsf.models.tirex.spec',
    'GlocalIB': 'moderntsf.models.glocalib.spec',
    'QuantileDLinear': 'moderntsf.models.quantile_dlinear.spec',
    'QuantilePatchTST': 'moderntsf.models.quantile_patchtst.spec',
    'MQRNN': 'moderntsf.models.mqrnn.spec',
    'GaussianMLP': 'moderntsf.models.gaussian_mlp.spec',
    'LatentTSF': 'moderntsf.models.latenttsf.spec',
    'CoRA': 'moderntsf.models.cora.spec',
    'DynamicTMoE': 'moderntsf.models.dynamic_tmoe.spec',
    'PULSE': 'moderntsf.models.pulse.spec',
    'OLinear': 'moderntsf.models.olinear.spec',
    'MAFS': 'moderntsf.models.mafs.spec',
    'TSRAG': 'moderntsf.models.tsrag.spec',
    'TimeMosaic': 'moderntsf.models.timemosaic.spec',
    'Kronos': 'moderntsf.models.kronos.spec',
    'MoFo': 'moderntsf.models.mofo.spec',
    'PHAT': 'moderntsf.models.phat.spec',
    'BiST': 'moderntsf.models.bist.spec',
    'MAGE': 'moderntsf.models.mage.spec',
    'STOP': 'moderntsf.models.stop.spec',
    'CauAir': 'moderntsf.models.cauair.spec',
    'AirCade': 'moderntsf.models.aircade.spec',
    'GTS': 'moderntsf.models.gts.spec',
    'STID': 'moderntsf.models.stid.spec',
    'GWNet': 'moderntsf.models.gwnet.spec',
    'D2STGNN': 'moderntsf.models.d2stgnn.spec',
    'DFDGCN': 'moderntsf.models.dfdgcn.spec',
    'STGCN': 'moderntsf.models.stgcn.spec',
    'AGCRN': 'moderntsf.models.agcrn.spec',
    'DCRNN': 'moderntsf.models.dcrnn.spec',
    'StemGNN': 'moderntsf.models.stemgnn.spec',
    'MTGNN': 'moderntsf.models.mtgnn.spec',
    'STGODE': 'moderntsf.models.stgode.spec',
    'STAEformer': 'moderntsf.models.staeformer.spec',
    'DGCRN': 'moderntsf.models.dgcrn.spec',
    'STDN': 'moderntsf.models.stdn.spec',
    'STPGNN': 'moderntsf.models.stpgnn.spec',
    'MegaCRN': 'moderntsf.models.megacrn.spec',
    'HimNet': 'moderntsf.models.himnet.spec',
    'STWave': 'moderntsf.models.stwave.spec',
    'BigST': 'moderntsf.models.bigst.spec',
    'ASTGCN': 'moderntsf.models.astgcn.spec',
    'GCLSTM': 'moderntsf.models.gclstm.spec',
    'DeepAir': 'moderntsf.models.deepair.spec',
    'STTN': 'moderntsf.models.sttn.spec',
    'GAGNN': 'moderntsf.models.gagnn.spec',
    'PM25_GNN': 'moderntsf.models.pm25gnn.spec',
    'AirFormer': 'moderntsf.models.airformer.spec',
    'DSTAGNN': 'moderntsf.models.dstagnn.spec',
    'PCDCNet': 'moderntsf.models.pcdcnet.spec',
    'AirPhyNet': 'moderntsf.models.airphynet.spec',
    'AirDualODE': 'moderntsf.models.airdualode.spec',
    'HL': 'moderntsf.models.hl.spec',
    'LSTM': 'moderntsf.models.lstm.spec',
    'RPMixer': 'moderntsf.models.rpmixer.spec',
    'MGSFformer': 'moderntsf.models.mgsfformer.spec',
    'CATS': 'moderntsf.models.cats.spec',
})

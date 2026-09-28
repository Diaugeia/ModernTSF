"""Flat metadata catalog for reusable model components.

The catalog documents shared implementation contracts without creating a
second model hierarchy.  It is metadata only: models import the concrete
component modules directly, so catalog discovery never changes runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ComponentSpec:
    """Identity and semantic boundary of one reusable component module."""

    name: str
    module: str
    contract: str
    public_symbols: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComponentMatch:
    """One retrieval candidate; semantic compatibility still needs review."""

    spec: ComponentSpec
    score: int
    matched_terms: tuple[str, ...]


class ComponentCatalog:
    """Flat lookup for shared component metadata."""

    def __init__(self, specs: tuple[ComponentSpec, ...]) -> None:
        self._specs = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("component names must be unique")

    def names(self) -> list[str]:
        return sorted(self._specs)

    def get(self, name: str) -> ComponentSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown component {name!r}") from exc

    def specs(self) -> tuple[ComponentSpec, ...]:
        return tuple(self._specs[name] for name in self.names())

    def match(self, query: str, limit: int = 5) -> tuple[ComponentMatch, ...]:
        """Rank lexical candidates without claiming semantic equivalence."""
        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not terms or limit < 1:
            return ()
        matches: list[ComponentMatch] = []
        for spec in self.specs():
            name_terms = set(re.findall(r"[a-z0-9]+", spec.name.lower()))
            symbol_terms = set(
                re.findall(r"[a-z0-9]+", " ".join(spec.public_symbols).lower())
            )
            keyword_terms = set(
                re.findall(r"[a-z0-9]+", " ".join(spec.keywords).lower())
            )
            contract_terms = set(re.findall(r"[a-z0-9]+", spec.contract.lower()))
            matched = terms & (
                name_terms | symbol_terms | keyword_terms | contract_terms
            )
            if not matched:
                continue
            score = sum(
                5
                if term in name_terms
                else 3
                if term in keyword_terms
                else 2
                if term in symbol_terms
                else 1
                for term in matched
            )
            matches.append(ComponentMatch(spec, score, tuple(sorted(matched))))
        matches.sort(key=lambda item: (-item.score, item.spec.name))
        return tuple(matches[:limit])


COMPONENT_CATALOG = ComponentCatalog(
    (
        ComponentSpec("adj_norm", "moderntsf.models._components.adj_norm", "Dense adjacency normalization.", keywords=("adjacency", "graph", "laplacian", "normalization")),
        ComponentSpec(
            "adain_style_norm",
            "moderntsf.models._components.adain_style_norm",
            "Adaptive instance normalization rescaling features to externally supplied statistics.",
            ("AdaptiveInstanceNorm1d",),
            ("adain", "adaptive", "non-stationary", "normalization", "style"),
        ),
        ComponentSpec(
            "adaptive_node_embedding_adjacency",
            "moderntsf.models._components.adaptive_node_embedding_adjacency",
            "Learnable node-embedding adaptive adjacency: softmax(relu(E1 @ E2^T)).",
            ("adaptive_node_embedding_adjacency",),
            ("adaptive", "adjacency", "embedding", "graph", "node", "softmax"),
        ),
        ComponentSpec(
            "channel_alignment",
            "moderntsf.models._components.channel_alignment",
            "Slice or zero-pad the trailing feature axis to a requested width.",
            ("fit_channels",),
            ("adapter", "channel", "feature", "padding", "shape"),
        ),
        ComponentSpec(
            "channel_wise_linear",
            "moderntsf.models._components.channel_wise_linear",
            "Shared or per-channel affine projection over the final sequence axis.",
            ("ChannelWiseLinear",),
            ("channel-wise", "forecast", "individual", "linear", "projection"),
        ),
        ComponentSpec(
            "mixer_block",
            "moderntsf.models._components.mixer_block",
            "Pre-normalized residual time mixing then residual feature mixing (TSMixer basic block).",
            ("MixerBlock",),
            ("feature", "gelu", "layernorm", "mixer", "residual", "time"),
        ),
        ComponentSpec(
            "dlinear",
            "moderntsf.models._components.dlinear",
            "Moving-average decomposition and channel-wise linear forecasting backbone.",
            ("DLinearBackbone",),
            ("decomposition", "linear", "moving-average", "seasonal", "trend"),
        ),
        ComponentSpec(
            "diffusion_conv",
            "moderntsf.models._components.diffusion_conv",
            "Graph-WaveNet diffusion concatenation and projection for static supports.",
            ("DiffusionConv2d",),
            ("diffusion", "graph", "graph-wavenet", "support", "spatiotemporal"),
        ),
        ComponentSpec("embed", "moderntsf.models._components.embed", "Value, position, calendar, patch, and inverted embeddings.", keywords=("calendar", "embedding", "patch", "position", "token")),
        ComponentSpec(
            "gated_dilated_conv",
            "moderntsf.models._components.gated_dilated_conv",
            "Causal dilated padding plus the WaveNet gated activation unit.",
            ("causal_pad", "gated_dilated_conv"),
            ("causal", "dilated", "gate", "gated-activation", "wavenet"),
        ),
        ComponentSpec(
            "dominant_periods",
            "moderntsf.models._components.dominant_periods",
            "Top-k FFT period selection with per-sample amplitude weights for BLC tensors.",
            ("dominant_periods",),
            ("amplitude", "fft", "frequency", "period", "spectrum"),
        ),
        ComponentSpec(
            "flatten_forecast_head",
            "moderntsf.models._components.flatten_forecast_head",
            "Shared or channel-wise linear forecast head over two flattened feature axes.",
            ("FlattenForecastHead",),
            ("channel-wise", "flatten", "forecast", "head", "linear", "patch"),
        ),
        ComponentSpec(
            "forecast_embedding",
            "moderntsf.models._components.forecast_embedding",
            "Value projection plus normalized six-column raw-calendar embedding.",
            ("RawCalendarEmbedding", "ForecastEmbedding"),
            ("calendar", "covariate", "embedding", "forecast", "value"),
        ),
        ComponentSpec(
            "gaussian_parameter_head",
            "moderntsf.models._components.gaussian_parameter_head",
            "Independent Gaussian location/positive-scale parameter projection.",
            ("GaussianParameterHead",),
            ("distribution", "gaussian", "location", "probabilistic", "scale"),
        ),
        ComponentSpec(
            "gated_fusion",
            "moderntsf.models._components.gated_fusion",
            "Learnable sigmoid gate that convexly blends two equal-shaped embeddings.",
            ("GatedFusion",),
            ("fusion", "gate", "gated", "mixture", "sigmoid"),
        ),
        ComponentSpec("graph_utils", "moderntsf.models._components.graph_utils", "Graph supports, Laplacians, and Chebyshev bases.", keywords=("adjacency", "chebyshev", "graph", "laplacian", "support")),
        ComponentSpec(
            "graph_spectral",
            "moderntsf.models._components.graph_spectral",
            "Robust scaled-Laplacian and exact-order Chebyshev support construction.",
            ("scaled_laplacian", "chebyshev_polynomials", "chebyshev_supports"),
            ("adjacency", "chebyshev", "degenerate", "graph", "laplacian", "spectral"),
        ),
        ComponentSpec(
            "last_value_center",
            "moderntsf.models._components.last_value_center",
            "Detached last-observed-timestep centering and restoration for BLC histories.",
            ("center_on_last_value", "restore_last_value"),
            ("centering", "detach", "last-value", "level", "residual"),
        ),
        ComponentSpec("marks", "moderntsf.models._components.marks", "Canonical temporal-mark and spatiotemporal input adapters.", keywords=("calendar", "covariate", "spatiotemporal", "timestamp")),
        ComponentSpec(
            "mamba",
            "moderntsf.models._components.mamba",
            "Kernel-free selective state-space mixer, normalization, and residual block.",
            ("RMSNorm", "MambaBlock", "MambaResidualBlock"),
            ("mamba", "mixer", "rmsnorm", "ssm", "state-space"),
        ),
        ComponentSpec("masking", "moderntsf.models._components.masking", "Attention mask construction.", keywords=("attention", "causal", "mask")),
        ComponentSpec(
            "hyper_state_scan",
            "moderntsf.models._components.hyper_state_scan",
            "Kernel-free scalar-state selective scan and a 2-D grid state mixer.",
            ("diagonal_selective_scan", "GridStateMixer"),
            ("grid", "hyper-state", "mamba", "scan", "ssm", "state-space"),
        ),
        ComponentSpec(
            "patchtst",
            "moderntsf.models._components.patchtst",
            "Patch extraction, time-series Transformer encoding, and PatchTST backbone.",
            ("PatchTSTBackbone",),
            ("backbone", "channel-independent", "patch", "transformer"),
        ),
        ComponentSpec("positional_encoding", "moderntsf.models._components.positional_encoding", "Patch-transformer positional encodings.", keywords=("encoding", "patch", "position", "transformer")),
        ComponentSpec(
            "periodic_query_bank",
            "moderntsf.models._components.periodic_query_bank",
            "Learnable per-phase vector table gathered into phase-aligned windows.",
            ("PeriodicQueryBank",),
            ("cycle", "gather", "period", "phase", "query"),
        ),
        ComponentSpec(
            "quantile_head",
            "moderntsf.models._components.quantile_head",
            "Input-conditioned monotone quantile head with non-crossing outputs.",
            ("QuantileHead", "validate_quantile_levels", "DEFAULT_QUANTILE_LEVELS"),
            ("monotone", "non-crossing", "probabilistic", "quantile"),
        ),
        ComponentSpec("revin", "moderntsf.models._components.revin", "Reversible instance normalization.", ("RevIN",), ("denormalization", "instance", "normalization", "reversible")),
        ComponentSpec("self_attention_family", "moderntsf.models._components.self_attention_family", "Shared full and probabilistic attention layers.", keywords=("attention", "full", "probabilistic")),
        ComponentSpec(
            "soft_tree",
            "moderntsf.models._components.soft_tree",
            "Differentiable binary and level-wise-shared tree routing with leaf interpolation.",
            ("SoftDecisionTree", "SoftObliviousTree", "binary_routes"),
            ("decision", "ensemble", "leaf", "oblivious", "routing", "soft", "tree"),
        ),
        ComponentSpec(
            "series_decomposition",
            "moderntsf.models._components.series_decomposition",
            "Edge-padded moving average and residual/trend decomposition for BLC data.",
            ("EdgePaddedMovingAverage", "SeriesDecomposition"),
            ("decomposition", "moving-average", "residual", "smoothing", "trend"),
        ),
        ComponentSpec(
            "topk_expert_router",
            "moderntsf.models._components.topk_expert_router",
            "Two-layer gating MLP with optional trainable noise and floor-blended top-k expert sparsification.",
            ("GatingMLP", "topk_dense_mix"),
            ("expert", "gate", "gating", "mixture", "moe", "routing", "sparse", "top-k"),
        ),
        ComponentSpec("transformer_encdec", "moderntsf.models._components.transformer_encdec", "Shared Transformer encoder and decoder blocks.", keywords=("attention", "decoder", "encoder", "transformer")),
        ComponentSpec("tst_transformer", "moderntsf.models._components.tst_transformer", "Time-series Transformer encoder blocks.", keywords=("attention", "encoder", "time-series", "transformer")),
    )
)

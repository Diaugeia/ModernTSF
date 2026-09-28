"""Focused regression tests for the Agent-first catalog layers."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from moderntsf.benchmark.command_runtime import module_slug as cli_module_slug
from moderntsf.benchmark.catalog_metadata import model_records, read_front_matter
from moderntsf.benchmark.cli import main as cli_main
from moderntsf.benchmark.commands.check_registry import check as check_model_catalog
from moderntsf.benchmark.model_contracts import audit_model_contracts
from moderntsf.benchmark.model_cards import REQUIRED_SECTIONS, audit_model_card_body
from moderntsf.benchmark.verification.reference import compare_model_reference
from moderntsf.benchmark.resource_cards import audit_resource_cards, dataset_records
from moderntsf.benchmark.commands.new_model import (
    _model as scaffold_model,
    _module_slug as scaffold_module_slug,
    _package_init as scaffold_package_init,
    _spec as scaffold_spec,
)
from moderntsf.benchmark.commands.new_dataset import _schema_single as scaffold_dataset_schema
from moderntsf.benchmark.config.loader import validate_task_compatibility
from moderntsf.benchmark.registry.datasets import DATASET_REGISTRY, register_dataset_by_name
from moderntsf.benchmark.registry.models import MODEL_CATALOG
from moderntsf.benchmark.runner.model_io import call_forecaster, slice_prediction_target
from moderntsf.models._components.adj_norm import gcn_norm, transition_matrix
from moderntsf.benchmark.catalog.component_audit import audit_components, component_dependency_closure
from moderntsf.benchmark.catalog.components import COMPONENT_CATALOG
from moderntsf.models._components.channel_alignment import fit_channels
from moderntsf.models._components.channel_wise_linear import ChannelWiseLinear
from moderntsf.models._components.dominant_periods import dominant_periods
from moderntsf.models._components.diffusion_conv import DiffusionConv2d
from moderntsf.models._components.flatten_forecast_head import FlattenForecastHead
from moderntsf.models._components.forecast_embedding import ForecastEmbedding
from moderntsf.models._components.gaussian_parameter_head import GaussianParameterHead
from moderntsf.models._components.graph_spectral import chebyshev_polynomials, chebyshev_supports, scaled_laplacian
from moderntsf.models._components.graph_utils import adj_to_supports, cheb_poly, normalize_adj_mx
from moderntsf.models._components.marks import to_spatiotemporal
from moderntsf.models._components.quantile_head import QuantileHead, validate_quantile_levels
from moderntsf.models._components.revin import RevIN
from moderntsf.models._components.series_decomposition import (
    EdgePaddedMovingAverage,
    SeriesDecomposition,
)
from moderntsf.tsf_core.agent_assets import audit_agent_assets
from moderntsf.tsf_core.paths import is_packaged_root, repository_root, require_checkout


class RepositoryContractTests(unittest.TestCase):
    def test_repository_resources_resolve_to_the_checkout(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(repository_root(), root)
        self.assertFalse(is_packaged_root())
        self.assertEqual(require_checkout("test"), root)

    def test_model_cards_do_not_use_the_obsolete_adapter_boilerplate(self) -> None:
        root = Path(__file__).resolve().parents[1]
        obsolete = "implementation/adapter: `model.py`"
        offenders = [
            str(path.relative_to(root))
            for path in (root / "src" / "moderntsf" / "models").glob("*/README.md")
            if obsolete in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(offenders, [])

    def test_every_model_directory_is_an_explicit_python_package(self) -> None:
        root = Path(__file__).resolve().parents[1]
        missing = [
            str(path.relative_to(root))
            for path in sorted((root / "src" / "moderntsf" / "models").glob("*/README.md"))
            if not (path.parent / "__init__.py").is_file()
        ]
        self.assertEqual(missing, [])

    def test_dataset_default_uses_the_ignored_local_data_layer(self) -> None:
        from moderntsf.benchmark.config.schema.dataset import DatasetConfig

        fields = DatasetConfig.model_fields
        self.assertEqual(fields["path"].default, "")
        self.assertNotIn("root_path", fields)
        self.assertNotIn("data_path", fields)
        with self.assertRaises(ValueError):
            DatasetConfig.model_validate(
                {"name": "custom", "root_path": "dataset", "data_path": "x.csv"}
            )

    def test_dataset_storage_contract_resolves_path_and_optional_id(self) -> None:
        for name in ("weather", "synthetic_st", "gift_eval"):
            register_dataset_by_name(name)
        self.assertEqual(
            DATASET_REGISTRY.get("weather").resolve_location(
                "./dataset/weather/weather.csv", None
            ),
            ("./dataset/weather", "weather.csv"),
        )
        self.assertEqual(
            DATASET_REGISTRY.get("synthetic_st").resolve_location("", None),
            ("", ""),
        )
        self.assertEqual(
            DATASET_REGISTRY.get("gift_eval").resolve_location(
                "./dataset/gift_eval", "electricity/15T"
            ),
            ("./dataset/gift_eval", "electricity/15T"),
        )

    def test_task_mode_is_an_executable_dataset_model_contract(self) -> None:
        register_dataset_by_name("weather")
        register_dataset_by_name("synthetic_st")
        flat = DATASET_REGISTRY.get("weather")
        nodes = DATASET_REGISTRY.get("synthetic_st")
        linear = MODEL_CATALOG.get("Linear")
        graph = MODEL_CATALOG.get("AGCRN")
        validate_task_compatibility("time_series", flat, linear)
        validate_task_compatibility("spatiotemporal", nodes, graph)
        with self.assertRaisesRegex(ValueError, "dataset 'weather'"):
            validate_task_compatibility("spatiotemporal", flat, graph)
        with self.assertRaisesRegex(ValueError, "model 'AGCRN'"):
            validate_task_compatibility("time_series", flat, graph)

    def test_graph_spectral_supports_handle_degenerate_graphs(self) -> None:
        identity = np.eye(3, dtype=np.float32)
        scaled = scaled_laplacian(identity)
        self.assertTrue(np.isfinite(scaled).all())
        self.assertEqual(chebyshev_polynomials(scaled, 1).shape, (1, 3, 3))
        supports = chebyshev_supports(identity, 3)
        self.assertEqual(tuple(supports.shape), (3, 3, 3))
        self.assertTrue(torch.isfinite(supports).all())
        with self.assertRaises(ValueError):
            chebyshev_polynomials(scaled, 0)

    def test_python_modules_have_descriptions(self) -> None:
        root = Path(__file__).resolve().parents[1]
        missing = []
        for path in (root / "src").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if not ast.get_docstring(tree):
                missing.append(str(path.relative_to(root)))
        self.assertEqual(missing, [])

    def test_public_module_slug_is_consistent(self) -> None:
        for normalize in (cli_module_slug, scaffold_module_slug):
            self.assertEqual(normalize("AirFormer"), "airformer")
            self.assertEqual(normalize("S_Mamba"), "s_mamba")

    def test_model_scaffold_emits_complete_compilable_package_templates(self) -> None:
        package_init = scaffold_package_init("PaperModel")
        model = scaffold_model("PaperModel", [("width", "int", "32")], False)
        spec = scaffold_spec(
            "PaperModel",
            "paper_model",
            [("width", "int", "32")],
            "time_series",
            ("revin",),
        )
        compile(package_init, "__init__.py", "exec")
        compile(model, "model.py", "exec")
        compile(spec, "spec.py", "exec")
        self.assertIn("from .model import Model", package_init)
        self.assertIn("x_mark_enc=None", model)
        self.assertNotIn("*args", model)
        self.assertNotIn("**kwargs", model)
        self.assertNotIn("mask=None", model)
        self.assertIn("components=('revin',)", spec)
        self.assertIn('ConfigDict(extra="forbid")', spec)

    def test_dataset_scaffold_emits_a_strict_parameter_schema(self) -> None:
        schema = scaffold_dataset_schema("paper_data")
        compile(schema, "paper_data.py", "exec")
        self.assertIn("DatasetParameters", schema)

    def test_every_registered_model_has_the_canonical_forward_signature(self) -> None:
        self.assertEqual(check_model_catalog(), [])

    def test_cli_routes_lightweight_catalog_descriptions(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli_main(["component", "show", "quantile_head"]), 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["module"], "moderntsf.models._components.quantile_head")
        self.assertIn("quantile_dlinear", payload["consumers"])
        self.assertEqual(payload["card"], "src/moderntsf/models/_components/quantile_head/README.md")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli_main(["dataset", "list", "--json"]), 0)
        datasets = json.loads(output.getvalue())
        self.assertEqual(len(datasets), 89)
        self.assertTrue(all(record["card"].endswith("/README.md") for record in datasets))

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                cli_main(["dataset", "search", "electricity", "15t", "--json"]),
                0,
            )
        dataset_matches = json.loads(output.getvalue())
        self.assertEqual(dataset_matches[0]["name"], "gift_eval/electricity_15T")

        for resource in ("component", "dataset"):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(cli_main([resource, "audit"]), 0)
            self.assertIn("40 components" if resource == "component" else "89/89", output.getvalue())

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli_main(["model", "list", "--json"]), 0)
        records = json.loads(output.getvalue())
        self.assertEqual(len(records), 190)
        self.assertTrue(all(record["summary"] for record in records))

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                cli_main(
                    [
                        "model",
                        "search",
                        "exogenous",
                        "transformer",
                        "--json",
                    ]
                ),
                0,
            )
        search_results = json.loads(output.getvalue())
        self.assertEqual(search_results[0]["name"], "TimeXer")
        self.assertIn("exogenous", search_results[0]["matched_terms"])
        self.assertNotIn("adapter", search_results[0])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli_main(["model", "show", "Linear"]), 0)
        shown = json.loads(output.getvalue())
        self.assertEqual(shown["verification"]["status"], "passed")
        self.assertEqual(shown["blockers"], [])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                cli_main(
                    [
                        "component",
                        "match",
                        "patch",
                        "transformer",
                        "backbone",
                        "--json",
                    ]
                ),
                0,
            )
        matches = json.loads(output.getvalue())
        self.assertEqual(matches[0]["name"], "patchtst")
        self.assertTrue(matches[0]["review_required"])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli_main(["model", "audit", "--summary"]), 0)
        audit = json.loads(output.getvalue())
        self.assertEqual(audit["models"], 190)
        self.assertNotIn("implementation", audit)
        self.assertNotIn("failed_by_implementation", audit)
        self.assertEqual(audit["failed"], 0)
        self.assertEqual(audit["blockers"], {})
        self.assertEqual(audit["verification"], {"passed": 190})
        self.assertEqual(sum(audit["verification"].values()), 190)
        self.assertEqual(
            audit["complete_codebase"],
            sum(record["codebase"] is not None for record in model_records(Path(__file__).resolve().parents[1])),
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                cli_main(["model", "audit", "CATS", "BiST", "--json"]), 0
            )
        records = {record["name"]: record for record in json.loads(output.getvalue())}
        self.assertNotIn("implementation", records["CATS"])
        self.assertNotIn("usage", records["CATS"]["codebase"])
        self.assertEqual(records["CATS"]["codebase"]["missing"], [])
        self.assertEqual(records["CATS"]["verification"]["status"], "passed")
        self.assertEqual(records["CATS"]["blockers"], [])
        self.assertNotIn("implementation", records["BiST"])
        self.assertNotIn("usage", records["BiST"]["codebase"])
        self.assertEqual(records["BiST"]["verification"]["status"], "passed")
        self.assertEqual(records["BiST"]["blockers"], [])

    def test_model_cards_are_the_only_descriptive_metadata_source(self) -> None:
        root = Path(__file__).resolve().parents[1]
        records = model_records(root)
        self.assertEqual(len(records), 190)
        self.assertTrue(all("implementation" not in record for record in records))
        self.assertTrue(
            all(
                record["codebase"] is None
                or set(record["codebase"]) == {"url", "revision", "license"}
                for record in records
            )
        )
        required = {"name", "summary", "paper", "paper_title", "venue", "year"}
        source = {"code", "revision", "license"}
        cards = [
            card
            for card in (root / "src" / "moderntsf" / "models").glob("*/README.md")
            if not card.parent.name.startswith("_")
        ]
        for card in cards:
            header = read_front_matter(card)
            self.assertTrue(required <= set(header), card)
            self.assertFalse(any(isinstance(value, dict) for value in header.values()), card)
            present_source = source & set(header)
            self.assertIn(present_source, (set(), source), card)
        forbidden = {"paper", "source", "evidence", "deviations", "implementation"}
        for record in records:
            spec_path = root / str(record["spec_file"])
            tree = ast.parse(spec_path.read_text(encoding="utf-8"))
            spec_call = next(
                node.value
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "SPEC" for target in node.targets)
            )
            self.assertTrue(forbidden.isdisjoint({kw.arg for kw in spec_call.keywords}))

    def test_model_cards_have_canonical_evidence_preserving_bodies(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cards = sorted(
            card
            for card in (root / "src" / "moderntsf" / "models").glob("*/README.md")
            if not card.parent.name.startswith("_")
        )
        self.assertEqual(len(cards), 190)
        self.assertEqual(REQUIRED_SECTIONS[0], "Method overview")
        self.assertEqual(
            [problem for card in cards for problem in audit_model_card_body(card)],
            [],
        )

    def test_every_cataloged_component_and_dataset_has_a_current_card(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(audit_resource_cards(root), [])
        self.assertTrue((root / "src/moderntsf/models/_foundation/README.md").is_file())
        self.assertEqual(len(COMPONENT_CATALOG.names()), 40)
        self.assertEqual(len(dataset_records(root)), 89)
        self.assertEqual(
            len(list((root / "src/moderntsf/models/_components").glob("*/README.md"))),
            40,
        )
        self.assertEqual(
            len(list((root / "catalog" / "datasets").glob("**/README.md"))),
            89,
        )

    def test_agent_assets_are_canonical(self) -> None:
        self.assertEqual(audit_agent_assets(), [])

    def test_model_and_component_catalogs_are_consistent(self) -> None:
        self.assertEqual(check_model_catalog(), [])
        self.assertEqual(audit_components(), [])
        self.assertEqual(
            set(component_dependency_closure({"patchtst"})),
            {
                "flatten_forecast_head",
                "patchtst",
                "positional_encoding",
                "revin",
                "tst_transformer",
            },
        )

    def test_selected_model_contract_batch(self) -> None:
        self.assertEqual(
            audit_model_contracts(
                ["ETSformer", "MixLinear", "CRIB"], forward=True
            ),
            [],
        )

    def test_shared_adjacency_normalizers_are_finite(self) -> None:
        adjacency = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertTrue(np.isfinite(gcn_norm(adjacency)).all())
        self.assertTrue(np.isfinite(transition_matrix(adjacency)).all())
        forward, reverse = adj_to_supports(adjacency)
        self.assertEqual(forward.dtype, torch.float32)
        torch.testing.assert_close(
            forward, torch.from_numpy(transition_matrix(adjacency)).float()
        )
        torch.testing.assert_close(
            reverse, torch.from_numpy(transition_matrix(adjacency.T)).float()
        )
        identity = np.eye(3, dtype=np.float32)
        self.assertEqual(cheb_poly(identity, 1).shape, (1, 3, 3))
        self.assertTrue(np.isfinite(normalize_adj_mx(identity, "scalap")[0]).all())
        with self.assertRaises(ValueError):
            cheb_poly(identity, 0)
        with self.assertRaises(ValueError):
            normalize_adj_mx(identity, "unknown")

    def test_shared_spatiotemporal_adapter_shape(self) -> None:
        values = torch.randn(2, 12, 4)
        marks = torch.zeros(2, 12, 6)
        adapted = to_spatiotemporal(values, marks)
        self.assertEqual(adapted.shape, (2, 12, 4, 3))
        torch.testing.assert_close(adapted[..., 0], values)

    def test_shared_channel_alignment_and_forecast_embedding_contracts(self) -> None:
        values = torch.randn(2, 5, 3)
        torch.testing.assert_close(fit_channels(values, 2), values[..., :2])
        padded = fit_channels(values, 5)
        torch.testing.assert_close(padded[..., :3], values)
        torch.testing.assert_close(padded[..., 3:], torch.zeros_like(padded[..., 3:]))
        with self.assertRaises(ValueError):
            fit_channels(values, 0)

        embedding = ForecastEmbedding(3, 8, 0.0)
        marks = torch.zeros(2, 5, 6)
        output = embedding(values, marks)
        self.assertEqual(output.shape, (2, 5, 8))
        with self.assertRaises(ValueError):
            embedding(values, marks[:, :-1])

    def test_repeated_model_helpers_are_extracted(self) -> None:
        root = Path(__file__).resolve().parents[1]
        forbidden = {"_fit_channels", "_levels", "ForecastEmbedding"}
        offenders = []
        for path in (root / "src" / "moderntsf" / "models").glob("*/model.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            }
            if names & forbidden:
                offenders.append(str(path.relative_to(root)))
        self.assertEqual(offenders, [])

    def test_revin_round_trip(self) -> None:
        values = torch.randn(2, 24, 3)
        revin = RevIN(3)
        restored = revin(revin(values, "norm"), "denorm")
        torch.testing.assert_close(restored, values, atol=1e-5, rtol=1e-5)
        subtract_last = RevIN(3, subtract_last=True)
        restored = subtract_last(subtract_last(values, "norm"), "denorm")
        torch.testing.assert_close(restored, values, atol=1e-5, rtol=1e-5)
        disabled = RevIN(3, enabled=False)
        self.assertIs(disabled(values, "norm"), values)
        self.assertIs(disabled(values, "denorm"), values)

    def test_edge_padded_series_decomposition_matches_reference(self) -> None:
        values = torch.randn(2, 9, 3, requires_grad=True)
        smoother = EdgePaddedMovingAverage(3)
        actual = smoother(values)
        padded = torch.cat([values[:, :1], values, values[:, -1:]], dim=1)
        expected = F.avg_pool1d(padded.permute(0, 2, 1), 3, stride=1).permute(
            0, 2, 1
        )
        torch.testing.assert_close(actual, expected)
        residual, trend = SeriesDecomposition(3)(values)
        torch.testing.assert_close(residual + trend, values)
        (residual.square().mean() + trend.square().mean()).backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        with self.assertRaisesRegex(ValueError, "positive odd"):
            EdgePaddedMovingAverage(4)

    def test_flatten_forecast_head_shared_and_individual_contracts(self) -> None:
        values = torch.randn(2, 3, 4, 5, requires_grad=True)
        shared = FlattenForecastHead(False, 3, 20, 7)
        shared_output = shared(values)
        torch.testing.assert_close(
            shared_output, shared.linear(values.flatten(start_dim=-2))
        )
        individual = FlattenForecastHead(True, 3, 20, 7)
        individual_output = individual(values)
        expected = torch.stack(
            [
                individual.linears[index](values[:, index].flatten(start_dim=-2))
                for index in range(3)
            ],
            dim=1,
        )
        torch.testing.assert_close(individual_output, expected)
        self.assertEqual(shared_output.shape, (2, 3, 7))
        self.assertEqual(individual_output.shape, (2, 3, 7))
        (shared_output.mean() + individual_output.mean()).backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_channel_wise_linear_matches_original_output_and_gradients(self) -> None:
        for individual in (False, True):
            actual_input = torch.randn(2, 3, 5, requires_grad=True)
            reference_input = actual_input.detach().clone().requires_grad_(True)
            projection = ChannelWiseLinear(5, 4, 3, individual)
            actual = projection(actual_input)
            if individual:
                reference = torch.zeros(2, 3, 4)
                for index in range(3):
                    reference[:, index, :] = projection.linears[index](
                        reference_input[:, index, :]
                    )
            else:
                reference = projection.linear(reference_input)
            torch.testing.assert_close(actual, reference)
            actual_grad = torch.autograd.grad(actual.square().sum(), actual_input)[0]
            reference_grad = torch.autograd.grad(
                reference.square().sum(), reference_input
            )[0]
            torch.testing.assert_close(actual_grad, reference_grad)

    def test_dominant_periods_matches_timesnet_msgnet_reference(self) -> None:
        actual_input = torch.randn(3, 16, 4, requires_grad=True)
        reference_input = actual_input.detach().clone().requires_grad_(True)
        periods, weights = dominant_periods(actual_input, 3)

        spectrum = torch.fft.rfft(reference_input, dim=1)
        strength = abs(spectrum).mean(0).mean(-1)
        strength[0] = 0
        _, indices = torch.topk(strength, 3)
        indices = indices.detach().cpu().numpy()
        reference_periods = reference_input.shape[1] // indices
        reference_weights = abs(spectrum).mean(-1)[:, indices]

        np.testing.assert_array_equal(periods, reference_periods)
        torch.testing.assert_close(weights, reference_weights)
        actual_grad = torch.autograd.grad(weights.sum(), actual_input)[0]
        reference_grad = torch.autograd.grad(
            reference_weights.sum(), reference_input
        )[0]
        torch.testing.assert_close(actual_grad, reference_grad)

    def test_diffusion_conv_matches_explicit_support_expansion(self) -> None:
        values = torch.randn(2, 3, 4, 5, requires_grad=True)
        supports = [
            torch.eye(4),
            torch.tensor(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ),
        ]
        layer = DiffusionConv2d(3, 2, dropout=0.0, support_len=2, order=2)
        actual = layer(values, supports)
        expanded = [values]
        for support in supports:
            first = torch.einsum("ncvl,vw->ncwl", values, support).contiguous()
            second = torch.einsum("ncvl,vw->ncwl", first, support).contiguous()
            expanded.extend([first, second])
        expected = layer.mlp(torch.cat(expanded, dim=1))
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_gaussian_parameter_head_preserves_both_scale_formulas(self) -> None:
        values = torch.randn(2, 5, requires_grad=True)
        for transform in ("softplus", "log1pexp"):
            head = GaussianParameterHead(5, 3, eps=1e-6, scale_transform=transform)
            loc, scale = head(values)
            raw_scale = head.scale_layer(values)
            expected_scale = (
                F.softplus(raw_scale)
                if transform == "softplus"
                else torch.log(1 + torch.exp(raw_scale))
            ) + 1e-6
            torch.testing.assert_close(loc, head.loc_layer(values))
            torch.testing.assert_close(scale, expected_scale)
            self.assertTrue(torch.all(scale > 0))
            gradient = torch.autograd.grad(
                (loc + scale).sum(), values, retain_graph=True
            )[0]
            self.assertTrue(torch.isfinite(gradient).all())

    def test_extracted_component_catalog_surface(self) -> None:
        for name in (
            "dlinear",
            "diffusion_conv",
            "channel_wise_linear",
            "dominant_periods",
            "flatten_forecast_head",
            "gaussian_parameter_head",
            "mamba",
            "patchtst",
            "quantile_head",
            "series_decomposition",
        ):
            self.assertIn(name, COMPONENT_CATALOG.names())
            self.assertTrue(COMPONENT_CATALOG.get(name).contract)

    def test_quantile_head_is_monotone_and_differentiable(self) -> None:
        values = torch.randn(2, 8, 3, 4, requires_grad=True)
        head = QuantileHead([0.1, 0.5, 0.9], in_features=4)
        output = head(values)
        self.assertEqual(output.shape, (2, 8, 3, 3))
        self.assertTrue(torch.all(output[..., 1:] >= output[..., :-1]))
        output.mean().backward()
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertEqual(validate_quantile_levels(None)[4], 0.5)
        for invalid in ([], [0.0, 0.5], [0.5, 0.5], [0.9, 0.1]):
            with self.assertRaises(ValueError):
                validate_quantile_levels(invalid)

    def test_model_io_preserves_probabilistic_axis(self) -> None:
        output = torch.randn(2, 12, 4, 9)
        target = torch.randn(2, 16, 4)
        sliced_output, sliced_target = slice_prediction_target(output, target, 6, "MS")
        self.assertEqual(sliced_output.shape, (2, 6, 1, 9))
        self.assertEqual(sliced_target.shape, (2, 6, 1))

    def test_model_io_does_not_mask_internal_type_errors(self) -> None:
        class Broken(torch.nn.Module):
            def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
                raise TypeError("internal failure")

        values = torch.randn(1, 4, 2)
        with self.assertRaisesRegex(TypeError, "internal failure"):
            call_forecaster(Broken(), values, None, values, None)

    def test_reference_comparison_checks_outputs_and_gradients(self) -> None:
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(4, 3)

            def forward(self, values):
                return torch.tanh(self.projection(values))

        official_reference = Block()
        local = Block()
        report = compare_model_reference(
            local,
            official_reference,
            (torch.randn(2, 5, 4),),
            module_map={"projection": "projection"},
        )
        self.assertTrue(report.passed)
        self.assertTrue(report.to_dict()["modes"]["train"]["passed"])


if __name__ == "__main__":
    unittest.main()

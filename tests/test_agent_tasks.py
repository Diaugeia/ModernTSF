"""Contract tests for provider-neutral Agent task templates."""

from __future__ import annotations

import unittest

from tsflab.benchmark.cli import main
from tsflab.tsf_core.agent_tasks import (
    AgentTaskError,
    audit_tasks,
    list_tasks,
    load_task,
    render_task,
)


class AgentTaskTests(unittest.TestCase):
    def test_catalog_is_valid_and_contains_bounded_workflows(self) -> None:
        self.assertEqual(audit_tasks(), [])
        self.assertEqual(
            {record["name"] for record in list_tasks()},
            {
                "autoresearch",
                "catalog-expansion",
                "component-curation",
                "experiment",
                "paper-reproduction",
                "paper-to-model",
                "paper-watch",
                "repo-final-audit",
                "verification-backlog",
            },
        )

    def test_render_binds_inputs_and_preserves_boundaries(self) -> None:
        payload = render_task(
            "autoresearch",
            {"question": "Does RevIN improve PatchTST?", "max_runs": "4"},
        )
        self.assertIn("Does RevIN improve PatchTST?", payload["prompt"])
        self.assertIn("4 total runs", payload["prompt"])
        self.assertEqual(payload["budget"]["max_runs"], 4)
        self.assertEqual(payload["permissions"]["model_code"], "no-change-without-separate-authorization")
        self.assertEqual(payload["skills"], ["run-autoresearch"])

    def test_component_curation_supports_bounded_periodic_scans(self) -> None:
        payload = render_task("component-curation", {})
        self.assertIn("repository-wide repeated implementation scan", payload["prompt"])
        self.assertEqual(payload["budget"]["max_component_extractions"], 2)
        self.assertEqual(
            payload["permissions"]["repository"],
            "write-components-affected-models-and-tests",
        )

    def test_every_template_has_a_directly_renderable_demo(self) -> None:
        for record in list_tasks():
            task = load_task(record["name"])
            supplied = {
                key: "1" if "maximum" in spec else "demo"
                for key, spec in task["inputs"].items()
                if spec.get("required", False)
            }
            payload = render_task(record["name"], supplied)
            self.assertTrue(payload["prompt"])
            self.assertEqual(payload["task"], record["name"])
            self.assertTrue(payload["permissions"])
            self.assertTrue(payload["budget"])

    def test_missing_or_unknown_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(AgentTaskError, "missing required"):
            render_task("paper-to-model", {"paper_url": "https://arxiv.org/abs/1"})
        with self.assertRaisesRegex(AgentTaskError, "unknown input"):
            render_task("paper-watch", {"surprise": "write everything"})
        with self.assertRaisesRegex(AgentTaskError, "between 1 and 12"):
            render_task("autoresearch", {"question": "test", "max_runs": "13"})

    def test_paper_to_model_prompt_has_an_explicit_preimplementation_gate(self) -> None:
        payload = render_task(
            "paper-to-model",
            {"paper_url": "https://arxiv.org/abs/1", "model_name": "Example"},
        )
        self.assertIn("Before writing code, confirm", payload["prompt"])
        self.assertNotIn("authorized..", payload["prompt"])

    def test_numeric_inputs_narrow_machine_readable_budgets(self) -> None:
        cases = [
            ("experiment", {"question": "test", "max_runs": "2"}, "max_runs", 2),
            (
                "paper-reproduction",
                {
                    "paper_url": "https://arxiv.org/abs/1",
                    "target": "reported primary table",
                    "max_runs": "3",
                },
                "max_runs",
                3,
            ),
            ("paper-watch", {"limit": "4"}, "max_candidates", 4),
            ("catalog-expansion", {"candidate_limit": "2"}, "max_candidates", 2),
            ("verification-backlog", {"batch_size": "1"}, "max_models", 1),
        ]
        for name, supplied, key, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(render_task(name, supplied)["budget"][key], expected)

    def test_cli_routes_task_validation(self) -> None:
        self.assertEqual(main(["agent", "task", "validate"]), 0)


if __name__ == "__main__":
    unittest.main()

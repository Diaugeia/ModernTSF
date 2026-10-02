"""Contract tests for provider-neutral Agent task templates."""

from __future__ import annotations

import unittest

from tsflab.cli.main import main
from tsflab.agent.tasks import (
    AgentTaskError,
    audit_tasks,
    list_tasks,
    load_task,
    render_task,
)


class ModuleMapTests(unittest.TestCase):
    def test_every_skill_and_task_has_exactly_one_module(self) -> None:
        from tsflab.agent.assets import audit_module_map
        from tsflab.agent import assets
        from tsflab.agent.modules import ALL_MODULES, CHAIN, owners

        on_disk = {path.parent.name for path in assets.SKILLS.glob("*/SKILL.md")}
        self.assertEqual(audit_module_map(on_disk), [])
        self.assertTrue(all(len(modules) == 1 for modules in owners("skills").values()))
        self.assertEqual(ALL_MODULES, (*CHAIN, "maintenance"))
        self.assertIn("unmapped-skill", " ".join(audit_module_map(on_disk | {"unmapped-skill"})))
        self.assertIn("ghost", " ".join(audit_module_map(on_disk - {"audit"} | {"ghost"})))

    def test_index_is_generated_from_the_map(self) -> None:
        from tsflab.agent import assets
        from tsflab.agent.index import render_index

        self.assertEqual(assets.INDEX.read_text(encoding="utf-8"), render_index(assets.ROOT / ".agents"))


class AgentTaskTests(unittest.TestCase):
    def test_catalog_is_valid_and_contains_bounded_workflows(self) -> None:
        self.assertEqual(audit_tasks(), [])
        self.assertEqual(
            {record["name"] for record in list_tasks()},
            {
                "autoresearch",
                "contribution",
                "experiment",
                "intake",
                "maintenance",
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

    def test_maintenance_supports_bounded_periodic_scans(self) -> None:
        payload = render_task("maintenance", {"mode": "curation"})
        self.assertIn("repository-wide repeated implementation scan", payload["prompt"])
        self.assertEqual(payload["budget"]["max_component_extractions"], 2)
        self.assertEqual(
            payload["permissions"]["repository"],
            "write-selected-models-components-tests-evidence-and-generated-projections",
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
            render_task("contribution", {})
        with self.assertRaisesRegex(AgentTaskError, "unknown input"):
            render_task("intake", {"surprise": "write everything"})
        with self.assertRaisesRegex(AgentTaskError, "between 1 and 12"):
            render_task("autoresearch", {"question": "test", "max_runs": "13"})

    def test_intake_is_read_only_by_default_with_a_preimplementation_gate(self) -> None:
        payload = render_task(
            "intake", {"paper_url": "https://arxiv.org/abs/1", "model_name": "Example"}
        )
        self.assertIn("before writing code, confirm", payload["prompt"])
        self.assertIn("Read-only", payload["inputs"]["approval"])
        self.assertEqual(payload["budget"]["max_models"], 1)

    def test_contribution_defaults_to_draft_only(self) -> None:
        payload = render_task("contribution", {"target": "#12"})
        self.assertIn("#12", payload["prompt"])
        self.assertIn("Draft only", payload["inputs"]["authorization"])
        self.assertEqual(payload["permissions"]["external_actions"], "none-unless-authorized")

    def test_numeric_inputs_narrow_machine_readable_budgets(self) -> None:
        cases = [
            ("experiment", {"question": "test", "max_runs": "2"}, "max_runs", 2),
            (
                "experiment",
                {
                    "question": "reproduce the reported primary table",
                    "paper_url": "https://arxiv.org/abs/1",
                    "max_runs": "3",
                },
                "max_runs",
                3,
            ),
            ("intake", {"candidate_limit": "4"}, "max_candidates", 4),
            ("maintenance", {"batch_size": "1"}, "max_models", 1),
        ]
        for name, supplied, key, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(render_task(name, supplied)["budget"][key], expected)

    def test_cli_routes_task_validation(self) -> None:
        self.assertEqual(main(["agent", "task", "validate"]), 0)


if __name__ == "__main__":
    unittest.main()

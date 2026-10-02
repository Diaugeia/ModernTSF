"""Step selection for ``tsf repo check`` (never runs the suite itself)."""

from __future__ import annotations

import unittest

from tsflab.cli.commands.gate import affected_smoke_configs, select_steps
from tsflab.core.paths import repository_root

ROOT = repository_root()


def names(scope: str, changed: list[str]) -> list[str]:
    return [step.name for step in select_steps(scope, changed, ROOT)]


class GateSelectionTests(unittest.TestCase):
    def test_full_scope_has_every_check_and_no_smoke(self) -> None:
        selected = names("full", [])
        for required in ("schema-export", "agent-assets", "model-cards",
                         "verification-stale", "dataset-cards", "component-cards",
                         "repo-audit", "web-submissions", "pytest"):
            self.assertIn(required, selected)
        self.assertNotIn("smoke-affected", selected)
        self.assertEqual(selected[-1], "pytest")  # slowest step last

    def test_changed_scope_without_model_code_skips_smoke(self) -> None:
        self.assertNotIn("smoke-affected", names("changed", ["docs/en/models.md"]))

    def test_runner_change_uses_representative_set(self) -> None:
        configs = affected_smoke_configs(["src/tsflab/experiments/runner/run_one.py"], ROOT)
        self.assertEqual(len(configs), 3)

    def test_model_change_maps_to_its_smoke_config(self) -> None:
        configs = affected_smoke_configs(["src/tsflab/models/crib/model.py"], ROOT)
        self.assertEqual(configs, ["configs/runs/smoke_crib.toml"])
        self.assertIn("smoke-affected",
                      names("changed", ["src/tsflab/models/crib/model.py"]))

    def test_model_without_smoke_config_falls_back(self) -> None:
        configs = affected_smoke_configs(["src/tsflab/models/nonexistent_x/model.py"], ROOT)
        self.assertEqual(len(configs), 3)

    def test_edited_smoke_config_is_selected(self) -> None:
        self.assertEqual(affected_smoke_configs(["configs/runs/smoke_crib.toml"], ROOT),
                         ["configs/runs/smoke_crib.toml"])


if __name__ == "__main__":
    unittest.main()

"""Regression tests for the repository's strict executable model gate."""

from __future__ import annotations

import unittest

from moderntsf.benchmark.model_contracts import audit_model_contracts


class StrictModelContractTests(unittest.TestCase):
    def test_strict_gate_covers_sequence_and_graph_models(self) -> None:
        failures = audit_model_contracts(
            names=["Linear", "GWNet"],
            strict=True,
        )
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()

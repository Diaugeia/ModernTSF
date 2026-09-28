#!/usr/bin/env python3
"""Regenerate canonical component and dataset README cards."""

from __future__ import annotations

from moderntsf.benchmark.resource_cards import write_resource_cards
from moderntsf.tsf_core.paths import require_checkout


def main() -> int:
    """Write cards only in a mutable source checkout."""
    root = require_checkout("generate resource cards")
    count = write_resource_cards(root)
    print(f"Generated {count} resource cards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

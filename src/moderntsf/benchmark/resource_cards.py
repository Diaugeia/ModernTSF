"""Generate and audit the canonical README cards for datasets and models._components.

Dataset card rendering and audit live in ``dataset_cards``; component card
rendering and audit live in ``component_cards``. Both mix a curated part with a
generated block, so this module only fans out to them.
"""

from __future__ import annotations

from pathlib import Path

from moderntsf.benchmark.catalog.components import COMPONENT_CATALOG
from moderntsf.benchmark.component_cards import (  # noqa: F401  (re-exported public names)
    audit_component_cards,
    component_card_path,
    update_component_card,
)
from moderntsf.benchmark.dataset_cards import (  # noqa: F401  (re-exported public names)
    DatasetRecord,
    audit_dataset_cards,
    dataset_card_path,
    dataset_records,
    expected_dataset_cards,
    write_dataset_cards,
)


def write_resource_cards(root: Path) -> int:
    """Refresh the generated blocks of every dataset and component card; return the count."""
    for spec in COMPONENT_CATALOG.specs():
        update_component_card(root, spec)
    return len(COMPONENT_CATALOG.names()) + write_dataset_cards(root)


def audit_resource_cards(root: Path) -> list[str]:
    """Report missing, stale, incomplete, or orphaned dataset and component cards."""
    errors = audit_component_cards(root) + audit_dataset_cards(root)
    expected = set(expected_dataset_cards(root))
    for path in sorted(set((root / "catalog" / "datasets").glob("**/README.md")) - expected):
        errors.append(f"orphaned resource card: {path.relative_to(root)}")
    components = {component_card_path(root, name) for name in COMPONENT_CATALOG.names()}
    actual = set((root / "src" / "moderntsf" / "models" / "_components").glob("*/README.md"))
    for path in sorted(actual - components):
        errors.append(f"orphaned resource card: {path.relative_to(root)}")
    return errors

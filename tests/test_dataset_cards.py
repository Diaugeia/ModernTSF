"""Dataset cards separate curated facts from the generated runtime block."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from tsflab.benchmark.cards.datasets import (
    CURATED_KEYS,
    END,
    REQUIRED_PRESET_KEYS,
    START,
    audit_dataset_cards,
    dataset_card_path,
    dataset_facts,
    dataset_records,
    family_names,
    read_dataset_facts,
    render_dataset_card,
    search_text,
    split_card,
    write_dataset_cards,
)

ROOT = Path(__file__).resolve().parents[1]


class DatasetCardContractTests(unittest.TestCase):
    def test_every_card_is_complete_current_and_free_of_placeholders(self) -> None:
        self.assertEqual(audit_dataset_cards(ROOT), [])

    def test_every_preset_card_carries_the_required_front_matter(self) -> None:
        for record in dataset_records(ROOT):
            front = read_dataset_facts(dataset_card_path(ROOT, record.name))
            missing = [key for key in REQUIRED_PRESET_KEYS if key not in front]
            self.assertEqual(missing, [], record.name)
            self.assertEqual(front["name"], record.name)
            self.assertEqual(front["task_modes"], list(record.task_modes))

    def test_family_cards_exist_for_selector_loaders(self) -> None:
        self.assertEqual(family_names(dataset_records(ROOT)), ("gift_eval",))
        facts = read_dataset_facts(dataset_card_path(ROOT, "gift_eval"))
        self.assertEqual(facts["kind"], "dataset-family")

    def test_facts_feed_dataset_search_text(self) -> None:
        records = {record.name: record for record in dataset_records(ROOT)}
        facts = dataset_facts(ROOT, ["etth1"])["etth1"]
        text = search_text(records["etth1"], facts)
        self.assertIn("transformer", text)
        self.assertIn(str(facts["domain"]).casefold(), text)


class DatasetCardGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = next(item for item in dataset_records(ROOT) if item.name == "etth1")
        self.existing = dataset_card_path(ROOT, "etth1").read_text(encoding="utf-8")

    def test_render_is_idempotent(self) -> None:
        self.assertEqual(render_dataset_card(self.record, self.existing), self.existing)

    def test_regeneration_preserves_curated_content_and_rewrites_only_the_block(self) -> None:
        stale = self.existing.replace("Registry loader: `ETTh1`", "Registry loader: `stale`")
        stale = stale.replace('loader: "ETTh1"', 'loader: "stale"')
        self.assertNotEqual(stale, self.existing)
        self.assertEqual(render_dataset_card(self.record, stale), self.existing)

    def test_curated_edits_survive_regeneration(self) -> None:
        edited = self.existing.replace("## Overview\n\n", "## Overview\n\nCURATED-MARKER ", 1)
        edited = edited.replace('domain: "', 'domain: "marker-', 1)
        rendered = render_dataset_card(self.record, edited)
        self.assertIn("CURATED-MARKER", rendered)
        self.assertEqual(split_card(rendered)[0]["domain"], split_card(edited)[0]["domain"])

    def test_new_card_gets_todo_placeholders_that_the_audit_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "configs", root / "configs")
            (root / "catalog" / "datasets").mkdir(parents=True)
            write_dataset_cards(root)
            front = read_dataset_facts(dataset_card_path(root, "etth1"))
            self.assertEqual(front["summary"], "TODO")
            self.assertTrue(all(key in front for key in REQUIRED_PRESET_KEYS))
            errors = audit_dataset_cards(root)
            self.assertTrue(any("TODO" in error for error in errors))
            text = dataset_card_path(root, "etth1").read_text(encoding="utf-8")
            self.assertEqual(text.count(START), 1)
            self.assertEqual(text.count(END), 1)

    def test_audit_flags_missing_fields_and_bad_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "configs", root / "configs")
            shutil.copytree(ROOT / "catalog", root / "catalog")
            path = dataset_card_path(root, "etth1")
            text = path.read_text(encoding="utf-8")
            text = text.replace('license: "', 'licence: "', 1)
            text = text.replace('redistribution: "conditional"', 'redistribution: "maybe"')
            path.write_text(text, encoding="utf-8")
            errors = "\n".join(audit_dataset_cards(root))
            self.assertIn("missing front-matter field 'license'", errors)
            self.assertIn("unknown front-matter field 'licence'", errors)
            self.assertIn("redistribution must be one of", errors)

    def test_curated_key_order_is_declared_once(self) -> None:
        self.assertEqual(len(set(CURATED_KEYS)), len(CURATED_KEYS))


if __name__ == "__main__":
    unittest.main()

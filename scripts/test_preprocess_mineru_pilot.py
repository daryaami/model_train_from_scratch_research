#!/usr/bin/env python3
"""Fixtures for the MinerU preprocessing pilot."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import yaml

import preprocess_mineru_pilot as pilot

SAMPLE = """Уче-
ный писал $C_{2}H_{5}OH$ и связь C-H остаётся.

$$
метил-
этил
$$

$$
CH3

11
$$

<table><tr><td>A</td><td>$NaCl$</td></tr></table>

![]()

или

или

или

© 1991 перевод

© 1991 перевод

© 1991 перевод

97

<!-- page 4 was not decoded -->

**** #### .... ____ ++++ ==== **** #### .... ____ ++++ ==== **** #### .... ____ ++++ ==== ****

Повтор абзаца достаточно длинный чтобы попасть в точные копии внутри книги и сохраниться один раз.

Повтор абзаца достаточно длинный чтобы попасть в точные копии внутри книги и сохраниться один раз.

C-
H
"""

CONFIG = {
    "header_min_repeats": 3,
    "header_max_chars": 180,
    "garbage_min_chars": 80,
    "garbage_max_letter_ratio": 0.25,
    "dedup_min_chars": 40,
    "page_window_chars": 1800,
    "homoglyph_min_letters": 4,
    "chunk_min_chars": 200,
    "formula_samples": 30,
    "table_samples": 10,
}


class PreprocessFixtures(unittest.TestCase):
    def test_dehyphen_keeps_formula_and_table(self) -> None:
        result = pilot.process_document(SAMPLE, CONFIG)
        filtered = result["filtered"]
        self.assertIn("Ученый писал $C_{2}H_{5}OH$", filtered)
        self.assertIn("связь C-H остаётся", filtered)
        self.assertIn("метил-\nэтил", filtered)
        self.assertIn("CH3\n\n11", filtered)
        self.assertIn("<table><tr><td>A</td><td>$NaCl$</td></tr></table>", filtered)
        self.assertNotIn("![]()", filtered)
        self.assertNotIn("© 1991 перевод", filtered)
        self.assertNotRegex(filtered, r"(?m)^97$")
        self.assertEqual(filtered.count("\nили\n"), 3)
        self.assertNotIn("was not decoded", filtered)
        self.assertNotIn("****", filtered)
        self.assertEqual(filtered.count("точные копии"), 1)
        self.assertIn("C-\nH", filtered)
        self.assertGreater(result["clean"]["dehyphen_chars"], 0)
        self.assertEqual(result["filter"]["dedup_count"], 1)
        self.assertEqual(result["filter"]["undecoded_count"], 1)
        self.assertEqual(result["filter"]["garbage_count"], 1)

        source_formulas, _unmatched = pilot.extract_formulas(result["normalized"])
        filtered_formulas, _filtered_unmatched = pilot.extract_formulas(filtered)
        _marks, intact = pilot.mark_kept(source_formulas, filtered_formulas)
        self.assertTrue(intact)
        self.assertEqual(filtered_formulas, source_formulas)

        source_tables = pilot.extract_tables(result["normalized"])
        filtered_tables = pilot.extract_tables(filtered)
        _table_marks, tables_intact = pilot.mark_kept(source_tables, filtered_tables)
        self.assertTrue(tables_intact)
        self.assertEqual(filtered_tables, source_tables)

    def test_refuses_to_write_into_mineru(self) -> None:
        target = pilot.MINERU_DIR / "_pilot_write_guard_should_not_exist.md"
        with self.assertRaises(SystemExit):
            pilot.write_text(target, "no", [pilot.MINERU_DIR])
        self.assertFalse(target.exists())

    def test_run_keeps_source_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            output = root / "out"
            source.mkdir()
            name = "sample.md"
            payload = SAMPLE.encode("utf-8")
            (source / name).write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            config = {
                "version": "v001",
                "source_dir": str(source),
                "output_dir": str(output),
                "tokenizer": "tokenizers/fineweb2_rus_v001/chosen/tokenizer.json",
                **CONFIG,
                "books": [{"file": name, "reason": "fixture"}],
            }
            config_path = root / "preprocess.yaml"
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            pilot.run(config_path)
            self.assertEqual(hashlib.sha256((source / name).read_bytes()).hexdigest(), digest)
            self.assertEqual([path.name for path in source.iterdir()], [name])
            filtered = (output / "03_filtered" / name).read_text(encoding="utf-8")
            self.assertIn("$C_{2}H_{5}OH$", filtered)
            self.assertIn("<table><tr><td>A</td><td>$NaCl$</td></tr></table>", filtered)
            self.assertIn("Ученый", filtered)
            self.assertTrue((output / "00_source" / name).is_file())
            self.assertTrue((output / "report" / "summary.md").is_file())
            self.assertTrue((output / "report" / "formula_samples.md").is_file())
            self.assertTrue((output / "report" / "table_samples.md").is_file())
            self.assertFalse(pilot.MINERU_DIR in output.resolve().parents)


if __name__ == "__main__":
    unittest.main()

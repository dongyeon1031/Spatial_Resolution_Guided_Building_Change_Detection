"""Focused checks for the external-checkpoint evaluator and split contract."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/vendor"))
sys.path.insert(0, str(ROOT / "evaluation"))
from evaluate import checked_checkpoint, safe_names
from frozen_helpers import _sliding_window_starts, _predict_sliding_single, normalize_binary_mask
from changedetection.evaluation.binary import BinaryChangeEvaluator


def cli(*args):
    return subprocess.run([sys.executable, "-B", str(ROOT / "evaluation/evaluate.py"), *args],
                          capture_output=True, text=True, cwd=tempfile.gettempdir())


class EvaluationTests(unittest.TestCase):
    def test_geometry(self):
        for length, expected in ((754, [0, 256, 498]), (750, [0, 256, 494]),
                                 (1024, [0, 256, 512, 768]), (128, [0])):
            starts = _sliding_window_starts(length, 256, 256)
            self.assertEqual(starts, expected)
            covered = np.zeros(length, dtype=bool)
            for start in starts:
                covered[start:start + 256] = True
            self.assertTrue(covered.all())

    def test_tiling_preserves_values(self):
        torch.set_num_threads(2)
        for height, width in ((754, 750), (63, 71)):
            before = torch.arange(height * width, dtype=torch.float32).reshape(1, 1, height, width)
            after = before + 3
            model = lambda a, b: torch.cat((a, b), dim=1)
            output = _predict_sliding_single(model, before, after, 256, 256, 4)
            self.assertTrue(torch.equal(output, torch.cat((before, after), dim=1)))

    def test_metrics(self):
        evaluator = BinaryChangeEvaluator()
        evaluator.add_batch(np.array([[0, 0], [1, 1]]), np.array([[0, 1], [0, 1]]))
        values = evaluator.compute()
        for field in ("precision", "recall", "f1"):
            self.assertAlmostEqual(getattr(values, field), 0.5, places=6)
        self.assertAlmostEqual(values.iou, 1 / 3, places=6)
        self.assertTrue(np.array_equal(normalize_binary_mask([[0, 127], [128, 255]]), [[0, 0], [1, 1]]))

    def test_lists_have_prepared_inventory(self):
        with (ROOT / "splits/source_images.csv").open() as stream:
            ids = {row["prepared_id"] for row in csv.DictReader(stream)}
        for path in (ROOT / "splits/lists").glob("*.txt"):
            self.assertFalse(set(safe_names(path)) - ids, path.name)

    def test_unsafe_split_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.txt"
            for contents in ("", "a.png\na.png\n", "../a.png\n", "/a.png\n"):
                path.write_text(contents)
                with self.assertRaises(ValueError):
                    safe_names(path)

    def test_checkpoint_is_required(self):
        result = cli("--entry", "single70k_aihub", "--data-root", "unused", "--output", "unused")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--checkpoint", result.stderr)

    def test_external_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.pth"
            path.write_bytes(b"test-only bytes, not model weights")
            row = {"weight_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(checked_checkpoint(path, row), path.resolve())
            with self.assertRaises(ValueError):
                checked_checkpoint(path, {"weight_sha256": "0" * 64})

    def test_missing_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                checked_checkpoint(Path(directory) / "absent.pth", {"weight_sha256": "0" * 64})

    def test_unbound_and_unknown_entries_rejected(self):
        for entry, message in (("single_whu", "withheld"), ("not_a_paper_entry", "Unknown entry")):
            result = cli("--entry", entry, "--checkpoint", "unused", "--data-root", "unused", "--output", "unused")
            self.assertEqual(result.returncode, 2)
            self.assertIn(message, result.stderr)

    def test_profiles(self):
        profiles = json.loads((ROOT / "evaluation/profiles.json").read_text())
        for profile in profiles.values():
            self.assertFalse(profile["resizing"])
            self.assertEqual((profile["tile_size"], profile["stride"]), (256, 256))
            self.assertEqual(profile["decision"], "argmax")

    def test_release_scope(self):
        with (ROOT / "manifest.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 24)
        bound = [row for row in rows if row["status"] == "external_weights_required"]
        self.assertEqual(len(bound), 21)
        self.assertEqual(len({row["weight_file"] for row in bound}), 15)
        self.assertFalse((ROOT / "weights").exists())

    def test_continuation_selection(self):
        entries = json.loads((ROOT / "evaluation/continuations.json").read_text())
        expected = {"levirplus": 14500, "s2looking": 37500, "aihub": 64500, "whu": 41500}
        for dataset, step in expected.items():
            self.assertEqual(entries["single70k_" + dataset]["selected_step"], step)
        self.assertTrue(all(value["trained_through_step"] == 70000 for value in entries.values()))

    def test_describe_without_gpu(self):
        result = cli("--describe", "single70k_aihub")
        self.assertEqual(result.returncode, 0, result.stderr)
        description = json.loads(result.stdout)
        self.assertEqual(description["entry"]["f1_percent"], "80.69")
        self.assertEqual(description["entry"]["status"], "external_weights_required")
        self.assertEqual(description["profile"]["layout"], ["A", "B", "label"])

    def test_license_and_reproduction_scope(self):
        licenses = json.loads((ROOT / "evaluation/license_status.json").read_text())
        self.assertTrue(licenses["approved"])
        self.assertEqual(licenses["authored_code_license"], "Apache-2.0")
        self.assertEqual(licenses["split_metadata_license"], "CC-BY-4.0")
        release = json.loads((ROOT / "evaluation/release.json").read_text())
        self.assertFalse(release["weights_included"])
        self.assertFalse(release["full_manuscript_reproduction_certified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

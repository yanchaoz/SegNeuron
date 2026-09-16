"""CLI/input regression tests plus opt-in-required tests of the real ELF backend."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

from Postprocess import FRMC_post as post


ROOT = Path(__file__).resolve().parents[1]


class PostprocessTests(unittest.TestCase):
    def test_tiff_axes_are_not_silently_reinterpreted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "head.tif"
            tifffile.imwrite(path, np.zeros((4, 8, 9), dtype=np.float32),
                             photometric="minisblack", metadata={"axes": "YZX"})
            with self.assertRaisesRegex(ValueError, "ZYX"):
                post._load_volume(path)

    def test_help_does_not_import_elf(self):
        code = (
            "import sys; sys.modules['elf'] = None; "
            "from Postprocess.FRMC_post import main; main(['--help'])"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--ground-truth", result.stdout)

    def test_invalid_input_is_rejected_before_backend_load(self):
        valid = np.ones((3, 2, 8, 8), dtype=np.float32)
        invalid = [np.ones((2, 2, 8, 8)), np.empty((3, 0, 8, 8)),
                   np.full_like(valid, np.nan), np.full_like(valid, 1.01),
                   np.full_like(valid, -0.01), valid.astype(complex)]
        with patch.object(post, "_load_elf", side_effect=AssertionError("backend must not load")):
            for values in invalid:
                with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                    post.post_mc(values)
            for beta in [0, 1, -1, np.nan, np.inf, True, None, [0.25]]:
                with self.subTest(beta=beta), self.assertRaises(ValueError):
                    post.post_mc(valid, beta)

    def test_fragments_are_disjoint_across_slices_and_solver_zero_is_preserved(self):
        fragment_slice = np.array([[1, 1, 9], [1, 9, 9]], dtype=np.uint64)
        captured = {}

        def compute_rag(watershed):
            captured["watershed"] = watershed.copy()
            return SimpleNamespace(numberOfEdges=3, numberOfNodes=4)

        def project(rag, labels):
            return labels[captured["watershed"]]

        feats = SimpleNamespace(
            compute_rag=compute_rag,
            compute_affinity_features=lambda *a: np.full((3, 1), 0.5),
            compute_boundary_mean_and_length=lambda *a: np.ones((3, 2)),
            project_node_labels_to_pixels=project,
        )
        mc = SimpleNamespace(
            transform_probabilities_to_costs=lambda costs, **kw: costs,
            multicut_kernighan_lin=lambda *a: np.array([0, 7, 0, 7]),
        )
        ws = SimpleNamespace(distance_transform_watershed=lambda *a, **kw: (fragment_slice.copy(), 9))
        with patch.object(post, "_load_elf", return_value=(feats, mc, ws)):
            result = post.post_mc(np.full((3, 2, 2, 3), 0.8))
        self.assertEqual(set(np.unique(captured["watershed"][0])), {0, 1})
        self.assertEqual(set(np.unique(captured["watershed"][1])), {2, 3})
        self.assertEqual(result.dtype, np.uint32)
        np.testing.assert_array_equal(result[0], result[1])
        np.testing.assert_array_equal(np.unique(result), [1, 2])

    def test_unassigned_watershed_is_an_error(self):
        ws = SimpleNamespace(distance_transform_watershed=lambda *a, **kw: (np.zeros((8, 8), dtype=np.uint64), 0))
        with patch.object(post, "_load_elf", return_value=(None, None, ws)):
            with self.assertRaisesRegex(RuntimeError, "unassigned"):
                post.post_mc(np.ones((3, 1, 8, 8)))

    def test_cli_fusion_output_and_optional_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            affs = np.full((3, 2, 8, 8), 0.9, dtype=np.float32)
            foreground = np.full((2, 8, 8), 0.7, dtype=np.float32)
            labels = np.ones(foreground.shape, dtype=np.uint32)
            labels[:, :, 4:] = 2
            np.save(directory / "affs.npy", affs)
            tifffile.imwrite(directory / "head.tif", foreground, photometric="minisblack")
            tifffile.imwrite(directory / "gt.tif", labels, photometric="minisblack")
            base = ["--affinities", str(directory / "affs.npy"), "--boundaries", str(directory / "head.tif")]
            for extension, with_gt in [("tif", False), ("npy", True)]:
                output = directory / f"labels.{extension}"
                args = base + ["--output", str(output)]
                if with_gt:
                    args += ["--ground-truth", str(directory / "gt.tif")]
                with patch.object(post, "post_mc", return_value=labels) as solver, patch("builtins.print") as printed:
                    self.assertEqual(post.main(args), 0)
                np.testing.assert_array_equal(solver.call_args.args[0], np.minimum(affs, foreground[None]))
                self.assertEqual(solver.call_args.args[1], 0.25)
                np.testing.assert_array_equal(post._load_volume(output), labels)
                summary = json.loads(printed.call_args.args[0])
                self.assertEqual("metrics" in summary, with_gt)
                if with_gt:
                    self.assertEqual(summary["metrics"]["arand"], 0)
                    self.assertEqual(summary["metrics"]["voi"], 0)
                original_bytes = output.read_bytes()
                with patch.object(post, "post_mc", side_effect=AssertionError("must fail first")):
                    with self.assertRaises(SystemExit):
                        post.main(args)
                self.assertEqual(output.read_bytes(), original_bytes)

    def test_backend_failure_does_not_leave_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.save(directory / "affs.npy", np.full((3, 2, 8, 8), 0.8))
            tifffile.imwrite(directory / "head.tif", np.full((2, 8, 8), 0.8), photometric="minisblack")
            output = directory / "out.npy"
            with patch.object(post, "_load_elf", side_effect=RuntimeError("backend unavailable")):
                with self.assertRaises(SystemExit):
                    post.main(["--affinities", str(directory / "affs.npy"), "--boundaries",
                               str(directory / "head.tif"), "--output", str(output)])
            self.assertFalse(output.exists())

    def test_ground_truth_validation_and_sparse_labels(self):
        for invalid in [np.zeros((2, 3), dtype=np.uint32), np.full((2, 3), -1), np.ones((2, 3), dtype=float)]:
            with self.assertRaises(ValueError):
                post._validate_ground_truth(invalid, (2, 3))
        with self.assertRaisesRegex(ValueError, "shape"):
            post._validate_ground_truth(np.ones((2, 3), dtype=np.uint32), (3, 2))
        compact = post._validate_ground_truth(np.array([[0, 2**40], [0, 9]], dtype=np.uint64), (2, 2))
        np.testing.assert_array_equal(compact, [[0, 2], [0, 1]])

    def test_exclusive_write_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "labels.npy"
            post._write_labels(output, np.ones((2, 3, 4), dtype=np.uint32))
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                post._write_labels(output, np.zeros((2, 3, 4), dtype=np.uint32))
            self.assertEqual(output.read_bytes(), before)


class RealElfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            post._load_elf()
        except RuntimeError as exc:
            if os.environ.get("SEGNEURON_REQUIRE_ELF") == "1":
                raise
            raise unittest.SkipTest(str(exc)) from exc

    @staticmethod
    def affinities():
        # Two large regions separated by a strong x boundary, linked across z.
        affs = np.full((3, 3, 48, 64), 0.98, dtype=np.float32)
        affs[2, :, :, 31:34] = 0.02
        return affs

    def test_real_multicut_preserves_original_partition(self):
        specification = importlib.util.spec_from_file_location("legacy_postprocess", ROOT / "legacy/Postprocess/FRMC_post.py")
        legacy = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(legacy)
        affs = self.affinities()
        before = affs.copy()
        for beta in (0.1, 0.25, 0.5, 0.75):
            with self.subTest(beta=beta):
                original = legacy.post_mc(affs, beta)
                result = post.post_mc(affs, beta)
                pairs = np.unique(np.stack([original.ravel(), result.ravel()], axis=1), axis=0)
                self.assertEqual(len(pairs), len(np.unique(original)))
                self.assertEqual(len(pairs), len(np.unique(result)))
                self.assertEqual(result.shape, affs.shape[1:])
                self.assertEqual(result.dtype, np.uint32)
                self.assertGreater(result.min(), 0)
        np.testing.assert_array_equal(affs, before)

    def test_real_cli_without_ground_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            affinities = self.affinities()
            np.save(directory / "affs.npy", affinities)
            tifffile.imwrite(directory / "head.tif", np.ones(affinities.shape[1:], dtype=np.float32), photometric="minisblack")
            output = directory / "labels.tif"
            with patch("builtins.print") as printed:
                post.main(["--affinities", str(directory / "affs.npy"), "--boundaries",
                           str(directory / "head.tif"), "--output", str(output)])
            labels = tifffile.imread(output)
            self.assertEqual(labels.dtype, np.uint32)
            self.assertEqual(labels.shape, affinities.shape[1:])
            self.assertGreater(labels.min(), 0)
            self.assertNotIn("metrics", json.loads(printed.call_args.args[0]))


if __name__ == "__main__":
    unittest.main()

"""CPU-only correctness tests; no external data or downloaded weights needed."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import ModuleType

import numpy as np
import tifffile
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("segneuron_inference", ROOT / "Train_and_Inference" / "inference.py")
inference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inference)


class CoordinateModel(torch.nn.Module):
    """Different channels expose coordinate transposition, normalization and blending errors."""
    def __init__(self):
        super().__init__()
        self.register_buffer("fixture", torch.tensor(0))

    def forward(self, x):
        assert not self.training
        assert torch.is_inference_mode_enabled()
        return torch.cat((x, 1 - x, x / 2), dim=1), x * 0.75


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_reconstructs_small_odd_nonsquare_and_singleton_volumes(self):
        for shape in ((1, 1, 1), (18, 31, 71), (21, 131, 197), (20, 128, 128)):
            with self.subTest(shape=shape):
                raw = np.random.default_rng(123).integers(0, 256, size=shape, dtype=np.uint8)
                actual, boundary = inference.infer_volume(CoordinateModel(), raw)
                normalized = raw.astype(np.float32) / 255
                self.assertEqual(actual.shape, (3,) + shape)
                self.assertEqual(boundary.shape, shape)
                self.assertEqual(actual.dtype, np.float32)
                self.assertTrue(np.isfinite(actual).all())
                # Overlap sums use float32, as in the original Gaussian blend.
                np.testing.assert_allclose(actual, np.stack((normalized, 1 - normalized, normalized / 2)), atol=1e-6)
                np.testing.assert_allclose(boundary, normalized * 0.75, atol=1e-6)
                self.assertGreaterEqual(actual.min(), 0)
                self.assertLessEqual(actual.max(), 1)

    def test_tiff_and_npy_load_identically(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = np.arange(3 * 5 * 7, dtype=np.uint8).reshape(3, 5, 7)
            for suffix in (".npy", ".tif"):
                path = Path(directory) / ("raw" + suffix)
                if suffix == ".npy":
                    np.save(path, raw)
                else:
                    tifffile.imwrite(path, raw, photometric="minisblack")
                np.testing.assert_array_equal(inference.load_volume(path), raw)

    def test_rejects_invalid_raw_arrays(self):
        for raw in (np.zeros((4, 4), np.uint8), np.zeros((0, 4, 4), np.uint8), np.zeros((4, 4, 4), np.uint16)):
            with self.subTest(shape=raw.shape, dtype=raw.dtype):
                with self.assertRaises(ValueError):
                    inference.infer_volume(CoordinateModel(), raw)

    def test_rejects_rgb_and_time_series_tiff(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.tif"
            tifffile.imwrite(path, np.zeros((8, 9, 3), np.uint8), photometric="rgb")
            with self.assertRaisesRegex(ValueError, "grayscale"):
                inference.load_volume(path)
            tifffile.imwrite(path, np.zeros((4, 8, 9), np.uint8), photometric="minisblack", metadata={"axes": "TYX"})
            with self.assertRaisesRegex(ValueError, "grayscale"):
                inference.load_volume(path)

    def test_checkpoint_formats_and_dataparallel_prefix(self):
        source = torch.nn.Conv3d(1, 1, 1)
        state = source.state_dict()
        variants = [state, {"model_weights": {"module." + k: v for k, v in state.items()}}, {"state_dict": state}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            for checkpoint in variants:
                torch.save(checkpoint, path)
                target = torch.nn.Conv3d(1, 1, 1)
                inference.load_checkpoint(target, path)
                for name, tensor in target.state_dict().items():
                    torch.testing.assert_close(tensor, state[name])

    def test_checkpoint_mismatches_and_collisions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model = torch.nn.Conv3d(1, 1, 1)
            torch.save({"weight": torch.zeros(1)}, path)
            with self.assertRaises(RuntimeError):
                inference.load_checkpoint(model, path)
            torch.save({"weight": model.weight, "module.weight": model.weight}, path)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                inference.load_checkpoint(model, path)

    def test_nonfinite_model_output_is_rejected(self):
        class InvalidModel(CoordinateModel):
            def forward(self, x):
                affinity, boundary = super().forward(x)
                return affinity * float("nan"), boundary
        with self.assertRaisesRegex(ValueError, "finite probabilities"):
            inference.infer_volume(InvalidModel(), np.zeros((1, 2, 3), np.uint8))

    def test_cli_help_does_not_require_site_packages(self):
        result = subprocess.run([sys.executable, "-S", str(Path(inference.__file__)), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint", result.stdout)
        self.assertIn("--device", result.stdout)

    def test_cli_refuses_existing_output_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            raw, checkpoint = output / "raw.npy", output / "weights.pt"
            np.save(raw, np.zeros((1, 2, 3), np.uint8))
            torch.save({}, checkpoint)
            with self.assertRaises(SystemExit) as caught:
                inference.main(["--input", str(raw), "--checkpoint", str(checkpoint), "--output-dir", str(output)])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse((output / "inference.json").exists())

    def test_cli_writes_complete_outputs_and_provenance(self):
        # Exercise actual I/O and checkpoint loading with a small known model.
        fake_package = ModuleType("model")
        fake_module = ModuleType("model.Mnet")
        fake_module.MNet = lambda *args, **kwargs: CoordinateModel()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, checkpoint, output = root / "raw.npy", root / "weights.pt", root / "output"
            np.save(raw, np.full((1, 7, 11), 127, np.uint8))
            torch.save({"model_weights": CoordinateModel().state_dict()}, checkpoint)
            with mock.patch.dict(sys.modules, {"model": fake_package, "model.Mnet": fake_module}):
                inference.main(["--input", str(raw), "--checkpoint", str(checkpoint), "--output-dir", str(output)])
            affinity = np.load(output / "affinities.npy")
            boundary = tifffile.imread(output / "boundaries.tif")
            metadata = json.loads((output / "inference.json").read_text())
            self.assertEqual(affinity.shape, (3, 1, 7, 11))
            self.assertEqual(boundary.shape, (1, 7, 11))
            self.assertEqual(metadata["input"]["sha256"], inference.sha256(raw))
            self.assertEqual(metadata["checkpoint"]["sha256"], inference.sha256(checkpoint))
            self.assertEqual(metadata["affinities"]["offsets_zyx"], [[-1, 0, 0], [0, -1, 0], [0, 0, -1]])


if __name__ == "__main__":
    unittest.main()

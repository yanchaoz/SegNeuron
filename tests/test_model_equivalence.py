import importlib.util
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LEGACY_REFERENCE = ROOT / "legacy" / "reference"


def load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.inputs = torch.linspace(-1.0, 1.0, 8 * 32 * 32).reshape(1, 1, 8, 32, 32)

    def assert_model_equivalent(self, legacy_path, current_path, output_channels):
        legacy_module = load_file_module("legacy_model", LEGACY_REFERENCE / legacy_path)
        current_module = load_file_module("polished_model", ROOT / current_path)

        torch.manual_seed(2026)
        legacy_model = legacy_module.MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub").eval()
        torch.manual_seed(2026)
        polished_model = current_module.MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub").eval()

        self.assertEqual(list(legacy_model.state_dict()), list(polished_model.state_dict()))
        with torch.inference_mode():
            legacy_outputs = legacy_model(self.inputs)
            polished_outputs = polished_model(self.inputs)

        self.assertEqual(output_channels, tuple(output.shape[1] for output in polished_outputs))
        for legacy_output, polished_output in zip(legacy_outputs, polished_outputs):
            self.assertTrue(torch.equal(legacy_output, polished_output))

    def test_supervised_model_is_numerically_equivalent(self):
        self.assert_model_equivalent(
            "Train_and_Inference/model/Mnet.py",
            "Train_and_Inference/model/Mnet.py",
            (3, 1),
        )

    def test_pretraining_model_is_numerically_equivalent(self):
        self.assert_model_equivalent(
            "Pretrain/model/Mnet_pretrain.py",
            "Pretrain/model/Mnet_pretrain.py",
            (1, 1),
        )


if __name__ == "__main__":
    unittest.main()

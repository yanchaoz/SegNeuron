import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LegacyContractTests(unittest.TestCase):
    def test_frozen_legacy_manifest_and_reference_hashes(self):
        manifest = json.loads((ROOT / "legacy" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            "659ce323acf2c4cc62e697a03c5d90fc809aa9b7",
            manifest["commit"],
        )
        self.assertEqual("legacy/upstream-659ce323", manifest["legacy_branch"])
        for relative_path, expected_digest in manifest["reference_sha256"].items():
            reference = ROOT / "legacy" / "reference" / relative_path
            self.assertEqual(expected_digest, hashlib.sha256(reference.read_bytes()).hexdigest())

    def test_mirrored_utility_modules_do_not_drift(self):
        pretrain_utils = ROOT / "Pretrain" / "utils"
        supervised_utils = ROOT / "Train_and_Inference" / "utils"
        for pretrain_file in pretrain_utils.glob("*.py"):
            if pretrain_file.name == "show.py":
                continue
            supervised_file = supervised_utils / pretrain_file.name
            self.assertTrue(supervised_file.is_file(), pretrain_file.name)
            self.assertEqual(
                pretrain_file.read_bytes(),
                supervised_file.read_bytes(),
                pretrain_file.name,
            )

    def test_training_entry_points_no_longer_reference_undefined_validation_provider(self):
        pretrain = (ROOT / "Pretrain" / "pretrain.py").read_text(encoding="utf-8")
        supervised = (ROOT / "Train_and_Inference" / "supervised_train.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("return train_provider, valid_provider", pretrain)
        self.assertNotIn("return train_provider, valid_provider", supervised)
        self.assertIn("loop(cfg, train_provider, model, optimizer, init_iters, writer)", supervised)

    def test_pretrain_sampler_excludes_the_invalid_upper_endpoint(self):
        source = (ROOT / "Pretrain" / "pretrain_provider.py").read_text(encoding="utf-8")
        self.assertIn("random.randrange(len(self.dataset))", source)
        self.assertNotIn("random.randint(0, len(self.dataset))", source)

    def test_valid_configuration_is_accepted(self):
        module = load_module(
            "supervised_config_validation",
            ROOT / "Train_and_Inference" / "config_validation.py",
        )
        config = {
            "NAME": "SegNeuron",
            "MODEL": {
                "model_type": "superhuman",
                "pre_train": False,
                "pretrain_path": "",
                "continue_train": False,
                "continue_path": "",
            },
            "TRAIN": {
                "resume": False,
                "loss_func": "BCELoss",
                "total_iters": 1,
                "base_lr": 0.01,
                "end_lr": 0.0001,
                "batch_size": 1,
                "num_workers": 0,
                "if_cuda": False,
                "random_seed": 666,
                "freq_mix_prob": 0.25,
                "spa_mix_prob": 0.25,
            },
            "DATA": {"data_folder": "configured/data"},
        }
        self.assertIs(config, module.validate_config(config))

    def test_placeholder_paths_fail_fast(self):
        module = load_module(
            "pretrain_config_validation",
            ROOT / "Pretrain" / "config_validation.py",
        )
        config = {
            "NAME": "SegNeuron",
            "MODEL": {
                "model_type": "superhuman",
                "pre_train": False,
                "pretrain_path": "",
                "continue_train": False,
                "continue_path": "",
            },
            "TRAIN": {
                "resume": False,
                "total_iters": 1,
                "base_lr": 0.01,
                "end_lr": 0.0001,
                "batch_size": 1,
                "num_workers": 0,
                "if_cuda": False,
                "random_seed": 666,
            },
            "DATA": {"data_folder": "/***/***"},
        }
        with self.assertRaises(module.ConfigError):
            module.validate_config(config)


if __name__ == "__main__":
    unittest.main()

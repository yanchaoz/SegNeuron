"""Exercise research training plumbing without the full EMNeuron corpus or a GPU."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def smoke(kind):
    import random
    from unittest.mock import Mock, patch
    from addict import Dict
    import numpy as np
    import torch
    import yaml

    folder, name = (('Pretrain', 'pretrain') if kind == 'pretrain'
                    else ('Train_and_Inference', 'supervised_train'))
    sys.path.insert(0, str(ROOT / folder))
    module = __import__(name)
    with (ROOT / folder / 'config' / 'SegNeuron.yaml').open() as handle:
        cfg = Dict(yaml.safe_load(handle))
    assert cfg.MODEL.model_type != 'mala'

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, inputs):
            output = torch.sigmoid(inputs + self.bias)
            return (output if kind == 'pretrain' else output.repeat(1, 3, 1, 1, 1), output)

    inputs = torch.zeros((1, 1, 2, 8, 8))
    target = torch.ones_like(inputs)
    batch = ((inputs, target, target) if kind == 'pretrain' else
             (inputs, target.repeat(1, 3, 1, 1, 1), target, target))
    provider = Mock()
    provider.next.side_effect = [batch, batch]
    with patch.object(module, 'Provider', return_value=provider):
        assert module.load_dataset(cfg) is provider

    with tempfile.TemporaryDirectory() as directory:
        cfg.record_path = cfg.cache_path = cfg.save_path = directory
        cfg.TRAIN.total_iters = 2
        cfg.TRAIN.display_freq = cfg.TRAIN.valid_freq = 1
        cfg.TRAIN.save_freq = 2
        cfg.TRAIN.base_lr = cfg.TRAIN.end_lr = 0.01
        module.cfg = cfg
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        module.loop(cfg, provider, model, optimizer, 0, Mock())
        assert provider.next.call_count == 2
        assert float(model.bias.detach()) > 0
        assert (Path(directory) / '000001.png').is_file()
        checkpoint = torch.load(Path(directory) / 'model-000002.ckpt', weights_only=True)
        assert checkpoint['current_iter'] == 2

    if kind == 'pretrain':
        from pretrain_provider import Train
        dataset = Train.__new__(Train)
        dataset.dataset = [np.zeros((20, 128, 128), dtype=np.uint8)]
        dataset.crop_from_origin = [20, 128, 128]
        dataset.simple_aug = lambda arrays: arrays
        random.seed(0)
        np.random.seed(0)
        outputs = dataset[0]
        assert all(value.shape == (1, 20, 128, 128) for value in outputs)
        assert all(np.isfinite(value).all() for value in outputs)


class TrainingTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('addict'), 'Install requirements-training.txt')
    def test_research_entrypoints_load_optimize_render_and_save(self):
        for kind in ('pretrain', 'supervised'):
            with self.subTest(kind=kind):
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--smoke', kind],
                                        capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--smoke':
        smoke(sys.argv[2])
    else:
        unittest.main()

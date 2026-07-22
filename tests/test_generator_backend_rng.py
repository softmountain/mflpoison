import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from mflpoison.generators.kplus1.backends import KPlusOneBackend


class _GlobalNoiseGenerator(nn.Module):
    def forward(self, noise, labels, len_audio, len_video):
        del labels, len_audio, len_video
        frame_noise = torch.randn(noise.shape[0], 2, 1, device=noise.device)
        return noise[:, :1].unsqueeze(1), frame_noise


class GeneratorBackendRNGTest(unittest.TestCase):
    def _backend(self):
        backend = object.__new__(KPlusOneBackend)
        backend.device = torch.device("cpu")
        backend.config = SimpleNamespace(
            num_classes=3,
            audio_seq_len=1,
            video_seq_len=2,
            z_dim=2,
        )
        backend.generator = _GlobalNoiseGenerator()
        backend.checkpoint_path = "unused.pt"
        return backend

    def test_seed_controls_nested_generator_noise_without_polluting_global_rng(self):
        labels = torch.tensor([1, 1])
        backend = self._backend()

        torch.manual_seed(123)
        expected_next = torch.randn(4)
        torch.manual_seed(123)
        first = backend.generate(labels, seed=77)
        actual_next = torch.randn(4)
        second = backend.generate(labels, seed=77)

        self.assertTrue(torch.equal(first.features["audio"], second.features["audio"]))
        self.assertTrue(torch.equal(first.features["video"], second.features["video"]))
        self.assertTrue(torch.equal(actual_next, expected_next))


if __name__ == "__main__":
    unittest.main()

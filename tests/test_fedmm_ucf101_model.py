import unittest

import torch

from fed_multimodal.model import MMActionClassifier
from fed_multimodal.trainers.optimizer import FedProxOptimizer


class UCF101ModelCompatibilityTest(unittest.TestCase):
    @staticmethod
    def _model(attention, attention_name):
        return MMActionClassifier(
            num_classes=51,
            audio_input_dim=80,
            video_input_dim=1280,
            d_hid=8,
            n_filters=2,
            en_att=attention,
            att_name=attention_name,
            d_head=2,
        )

    def test_retained_attention_variants_forward_and_reload_state_dict(self):
        audio = torch.randn(2, 16, 80)
        video = torch.randn(2, 2, 1280)
        audio_lengths = torch.tensor([16, 16])
        video_lengths = torch.tensor([2, 2])
        for attention, name in (
            (False, "base"),
            (True, "base"),
            (True, "additive"),
            (True, "multihead"),
            (True, "fuse_base"),
        ):
            with self.subTest(attention=attention, name=name):
                model = self._model(attention, name).eval()
                state = model.state_dict()
                reloaded = self._model(attention, name).eval()
                reloaded.load_state_dict(state, strict=True)
                with torch.no_grad():
                    predictions, embedding = reloaded(
                        audio,
                        video,
                        audio_lengths,
                        video_lengths,
                    )
                self.assertEqual(tuple(predictions.shape), (2, 51))
                self.assertEqual(embedding.shape[0], 2)

    def test_invalid_enabled_attention_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "attention must be one of"):
            self._model(True, "hirarchical")

    def test_fedprox_optimizer_remains_available_to_client_fedavg(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = FedProxOptimizer([parameter], lr=0.1, mu=0.0)
        parameter.grad = torch.tensor([2.0])
        optimizer.step()
        self.assertTrue(torch.allclose(parameter, torch.tensor([0.8])))


if __name__ == "__main__":
    unittest.main()
